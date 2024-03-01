import json
import cv2
import numpy as np
import os , glob
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
import argparse
import datetime
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

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

        # if not isinstance(ext, (tuple, list, ListConfig)):
        #     ext = [ext]

        # with open(os.path.join(root_dir, 'test_paths.json')) as f:
        #     self.paths = json.load(f)

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
            target_im  = (target_im .astype(np.float32) / 127.5) - 1.0


        except:
            # very hacky solution, sorry about this
            filename = os.path.join(self.root_dir, '692db5f2d3a04bb286cb977a7dba903e_1')  # this one we know is valid
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
            # normalize
            canny_r = canny_r.astype(np.float32) / 255.0
            target_im  = (target_im .astype(np.float32) / 127.5) - 1.0



        data["img"] = target_im
        data["hint"] = canny_r
        data["camera_pose"] = self.get_T(target_RT, cond_RT) # actually the difference between two camera
        data["txt"] = prompt

        print("test prompt is ", prompt)
        print("img shape", target_im.shape, "hint shape", canny_r.shape)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))
#
#
#
# # setting for training
batch_size= 1
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
#
#
# dataset = ObjaverseDataModuleFromConfig(root_dir, batch_size, total_view,  num_workers)
# dataset.prepare_data()
# dataset.setup()
#
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader
# # from tutorial_dataset import MyDataset
# from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
# #
# #
# # Configs
# resume_path0 = './models/control_sd15_ini.ckpt'  # totorial
conv11 = 'models/control_v11_sd15_canny_full.ckpt'  # conv 1.1
# conv1 = 'models/control_sd15_canny.pth'  # conv 1
zero123 = 'zero123-xl.ckpt'

# tut = torch.load(resume_path0) # <class 'collections.OrderedDict'>
con11 = torch.load(conv11)   # <class 'dict'>
zero123 = torch.load(zero123) # <class 'collections.OrderedDict'>

# # # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
# model = create_model('models/yuch_v11p_sd15_canny.yaml').cpu()
# model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
# # model.load_state_dict(torch.load(resume_path))
# model.learning_rate = learning_rate
# model.sd_locked = sd_locked
# model.only_mid_control = only_mid_control
# print("load done
# tut_keys = list(tut.keys())
con_keys = list(con11.keys())
zero123_keys = list(zero123['state_dict'].keys())

print("v1.1 keys" , len(con_keys) ,con_keys)

print("\n\n\n zero123 keys:" , len(zero123_keys) ,zero123_keys )



intersection = []
# same_shape = 0
not_in_zero123 = []

for k in con_keys:
    if k in zero123_keys :
        intersection.append(k)

    else:
        not_in_zero123.append(k)

print("\n\n\n\nin \n", intersection)
print("\n\n\n\nnot \n" , not_in_zero123)

for k in intersection:
    if k[:21] == 'model.diffusion_model':
        print("add parameter of ", k, " from conv1.1 to conv 1 \n")
        if k == 'model.diffusion_model.input_blocks.0.0.weight':
            print(zero123['state_dict'][k].shape)
            con11[k] = zero123['state_dict'][k][:, :4, :, :]
            print(con11[k].shape)
        else:
            con11[k] = zero123['state_dict'][k]
    # print(con22[k] == con11[k])
#
print("saving the ckpt.")
torch.save(con11,'control_v11_zero123_canny.ckpt')
