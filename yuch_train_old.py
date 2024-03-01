import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from pathlib import Path
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from einops import rearrange
from torch.utils.data.distributed import DistributedSampler
import webdataset as wds
from omegaconf import DictConfig, ListConfig
import math
import matplotlib.pyplot as plt
import sys
from PIL import Image
import random
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view,  num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view
        image_transforms = [torchvision.transforms.Resize(256)]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        # total_view = 4
        # print("t1 train_data")
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                             sampler=sampler)

    def val_dataloader(self):
        # print("t1 val_data")
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        # print("t1 test_data")
        return wds.WebLoader(
            ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation), \
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.mean(image)
    # apply automatic Canny edge detection using the computed media

    lower = int(max(10, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    print(v, lower, upper)
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def random_canny(image):
    # compute the median of the single channel pixel intensities
    # apply automatic Canny edge detection using the computed media

    lower = 10 + np.random.random() * 90 # lower 0 ~ 100
    upper = 150 + np.random.random()*100 # upper 150 ~ 250

    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


class ObjaverseData(Dataset):
    def __init__(self,
                 root_dir='.objaverse/hf-objaverse-v1/views',
                 image_transforms=[],
                 ext="png",
                 default_trans=torch.zeros(3),
                 postprocess=None,
                 return_paths=False,
                 total_view=12,
                 validation=False
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths

        # if isinstance(postprocess, DictConfig):
        #     postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view
        self.bad_files = []

        with open('valid_paths.json') as f:
            self.paths = json.load(f)

        total_objects = len(self.paths)

        print("*********number of total objects", total_objects)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):]  # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)]  # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):

        data = {}
        total_view = self.total_view
        index_target, index_cond = random.sample(range(total_view), 2)  # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)

        # color = [1., 1., 1., 1.]

        try:
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            # read prompt from BLIP
            f = open(os.path.join(filename, "BLIP_best_text.txt") , 'r')
            prompt = f.readline()
            # get cond_im and target_im
            cond_im = cv2.imread(os.path.join(filename, '%03d.png' % index_cond))
            target_im = cv2.imread(os.path.join(filename, '%03d.png' % index_target))
            # test_im = cv2.imread("test.png")

            # print("*** cond_im.shape", cond_im.shape)
            # print("*** target_im.shape",target_im.shape)
            # print("*** test_im.shape",test_im.shape)

            # BGR TO RGB
            cond_im  = cv2.cvtColor(cond_im , cv2.COLOR_BGR2RGB)
            target_im  = cv2.cvtColor(target_im , cv2.COLOR_BGR2RGB)
            # get canny edge
            canny_r = random_canny(cond_im)
            # print("*** canny_r.shape", canny_r.shape)
            canny_r = canny_r[:,:,None]
            canny_r = np.concatenate([canny_r, canny_r, canny_r], axis=2)
            # print("*** canny_r.shape after concatenate", canny_r.shape)
            # normalize
            canny_r = canny_r.astype(np.float32) / 255.0
            canny_r = torch.tensor(canny_r)
            target_im  = (target_im .astype(np.float32) / 127.5) - 1.0
            target_im = torch.tensor(target_im)


        except:

            if filename not in self.bad_files:
                self.bad_files.append(filename)
            print("Bad file encoutered : ", filename)
            # very hacky solution, sorry about this
            filename = os.path.join(self.root_dir, '0a0b504f51a94d95a2d492d3c372ebe5')  # this one we know is valid
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            # read prompt from BLIP
            f = open(os.path.join(filename, "BLIP_best_text.txt") , 'r')
            prompt = f.readline()
            # get cond_im and target_im
            cond_im = cv2.imread(os.path.join(filename, '%03d.png' % index_cond))
            target_im = cv2.imread(os.path.join(filename, '%03d.png' % index_target))
            # BGR TO RGB
            cond_im  = cv2.cvtColor(cond_im , cv2.COLOR_BGR2RGB)
            target_im  = cv2.cvtColor(target_im , cv2.COLOR_BGR2RGB)
            # get canny edge
            canny_r = random_canny(cond_im)
            canny_r = canny_r[:, :, None]
            canny_r = np.concatenate([canny_r, canny_r, canny_r], axis=2)
            # normalize
            canny_r = canny_r.astype(np.float32) / 255.0
            target_im  = (target_im .astype(np.float32) / 127.5) - 1.0

            canny_r = torch.tensor(canny_r)
            target_im = torch.tensor(target_im)



        data["img"] = target_im
        data["hint"] = canny_r
        data["camera_pose"] = self.get_T(target_RT, cond_RT) # actually the difference between two camera
        data["txt"] = prompt

        # print("test prompt is ", prompt)
        # print("img shape", target_im.shape, "hint shape", canny_r.shape)


        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)




# setting for training
batch_size=15
gpus=1
# total batch = batch_size * gpus
root_dir = '/yuch_ws/views_release'
# setting for local test
# batch_size= 1
# gpus=1
# root_dir = 'objvarse_views'

num_workers = 16
total_view = 12
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


dataset = ObjaverseDataModuleFromConfig(root_dir, batch_size, total_view,  num_workers)
dataset.prepare_data()
dataset.setup()

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
#
#
# Configs
# resume_path0 = './models/control_sd15_ini.ckpt'  # totorial
resume_path = 'models/control_v11_sd15_canny_full.ckpt'  # conv 1.1
# resume_path = 'models/control_sd15_canny.pth'  # conv 1


# # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('models/yuch_v11p_sd15_canny.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
# model.load_state_dict(torch.load(resume_path))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
#
#
# # Misc
#
#dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

logger = ImageLogger(batch_frequency=logger_freq)

checkpoint_callback = ModelCheckpoint(monitor = 'global_step',dirpath = 'logs/checkpoints',
                                              filename = 'control_{epoch}-{step}',verbose=True,
                                              every_n_train_steps=500)
# plugins=[DDPPlugin(find_unused_parameters=True)]
# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
trainer = Trainer(plugins=[DDPPlugin()],accelerator='ddp',
                          accumulate_grad_batches=1, benchmark=True, gpus='0,',
                  num_sanity_val_steps=0, val_check_interval=5000000,callbacks=[logger,checkpoint_callback] )
# trainer = pl.Trainer(accelerator="ddp", devices='0,', precision=32, callbacks=[logger,checkpoint_callback])

# Train!
trainer.fit(model, dataset)