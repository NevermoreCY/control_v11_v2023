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
from cldm.logger import ImageLogger2
from packaging import version
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
import time
from pytorch_lightning.utilities import rank_zero_info
@rank_zero_only
def rank_zero_print(*args):
    print(*args)

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--finetune_from",
        type=str,
        nargs="?",
        default="",
        help="path to checkpoint to load model state from"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="resolution of image",
    )
    return parser

class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view,  num_workers=4,valid_path='valid_path.json', img_size=256, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view
        self.image_size = img_size
        self.valid_path = valid_path
        image_transforms = [torchvision.transforms.Resize(img_size)]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        # total_view = 4
        # print("t1 train_data")
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms, image_size=self.image_size,valid_path=self.valid_path)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                             sampler=sampler)

    def val_dataloader(self):
        # print("t1 val_data")
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms,image_size=self.image_size,valid_path=self.valid_path)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        # print("t1 test_data")
        return wds.WebLoader(
            ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation,image_size=self.image_size,valid_path=self.valid_path), \
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
                 validation=False,
                 image_size=256,
                 valid_path='valid_path.json'
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
        self.image_size = image_size

        self.bad_files = []

        # if not isinstance(ext, (tuple, list, ListConfig)):
        #     ext = [ext]

        # with open(os.path.join(root_dir, 'test_paths.json')) as f:
        #     self.paths = json.load(f)

        with open(valid_path) as f:
            self.paths = json.load(f)

        total_objects = len(self.paths)

        print("*********number of total objects", total_objects)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):]  # used last 1% as validation
        else:
            self.paths = self.paths[:]  # used all for training since this script is not doing validation, we do it after getting checkpoints
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
            f = open(os.path.join(filename, "BLIP_best_text_v2.txt") , 'r')
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
            # print("image size is ", self.image_size)
            cond_im = cv2.resize(cond_im, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            target_im = cv2.resize(target_im, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

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
            f = open(os.path.join(filename, "BLIP_best_text_v2.txt") , 'r')
            prompt = f.readline()
            # get cond_im and target_im
            cond_im = cv2.imread(os.path.join(filename, '%03d.png' % index_cond))
            target_im = cv2.imread(os.path.join(filename, '%03d.png' % index_target))
            # BGR TO RGB
            cond_im  = cv2.cvtColor(cond_im , cv2.COLOR_BGR2RGB)
            target_im  = cv2.cvtColor(target_im , cv2.COLOR_BGR2RGB)
            # resize
            cond_im = cv2.resize(cond_im, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            target_im = cv2.resize(target_im, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
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


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config,
                 lightning_config, debug):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug

    def on_keyboard_interrupt(self, trainer, pl_module):
        if not self.debug and trainer.global_rank == 0:
            rank_zero_print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)


            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            rank_zero_print("Project config")
            rank_zero_print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            rank_zero_print("Lightning config")
            rank_zero_print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))



class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, log_all_val=False):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_all_val = log_all_val

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if self.log_all_val and split == "val":
            should_log = True
        else:
            should_log = self.check_frequency(check_idx)
        if (should_log and  (check_idx % self.batch_freq == 0) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                rank_zero_print(e)
                pass
            return True
        return False

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    def on_train_batch_end(self, trainer, pl_module, outputs,batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    def on_validation_batch_end(self, trainer, pl_module,outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    # def on_train_epoch_end(self, trainer, pl_module, outputs):
    # https://github.com/williamFalcon/pytorch-lightning-vae/issues/7
    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class SingleImageLogger(Callback):
    """does not save as grid but as single images"""
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, log_always=False):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_always = log_always

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)
        for k in images:
            subroot = os.path.join(root, k)
            os.makedirs(subroot, exist_ok=True)
            base_count = len(glob.glob(os.path.join(subroot, "*.png")))
            for img in images[k]:
                if self.rescale:
                    img = (img + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                img = img.transpose(0, 1).transpose(1, 2).squeeze(-1)
                img = img.numpy()
                img = (img * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}_{:08}.png".format(
                    k,
                    global_step,
                    current_epoch,
                    batch_idx,
                    base_count)
                path = os.path.join(subroot, filename)
                Image.fromarray(img).save(path)
                base_count += 1

    def log_img(self, pl_module, batch, batch_idx, split="train", save_dir=None):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0) or self.log_always:
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir if save_dir is None else save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                rank_zero_print(e)
            return True
        return False
#
#
# # setting for training
# batch_size= 20
# gpus=1
# # total batch = batch_size * gpus
# root_dir = '/yuch_ws/views_release'
# # setting for local test
# # batch_size= 1
# # gpus=1
# # root_dir = 'objvarse_views'
#
# num_workers = 16
# total_view = 12
# logger_freq = 300
# learning_rate = 1e-5
# sd_locked = True
# only_mid_control = False
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
# # resume_path0 = './models/control_sd15_ini.ckpt'  # totorial
# # resume_path1 = './models/control_v11p_sd15_canny.pth'  # conv 1.1
# resume_path = 'models/control_sd15_canny.pth'  # conv 1
#
#
# # # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
# model = create_model('models/yuch_v11p_sd15_canny.yaml').cpu()
# model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
# # model.load_state_dict(torch.load(resume_path))
# model.learning_rate = learning_rate
# model.sd_locked = sd_locked
# model.only_mid_control = only_mid_control
# #
# #
# # # Misc
# #
# #dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
# logger = ImageLogger(batch_frequency=logger_freq)
# # trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
# trainer = pl.Trainer(accelerator="gpu", devices=gpus, strategy="ddp", precision=32, callbacks=[logger])
#
# # Train!
# trainer.fit(model, dataset)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    print("***opt is ", opt)
    print("***unkown is ", unknown)

    if opt.resume:

        # resume training
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            print("is file")
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])  # eg : logs/2023_08_01_training
            ckpt = opt.resume
        else:

            assert os.path.isdir(opt.resume), opt.resume
            print("is dir")
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        print("base_configs:", base_configs)
        opt.base = base_configs + opt.base
        print("opt.base", opt.base)
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        # first time training
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    imgdir = os.path.join(logdir, "images_control")

    # os.makedirs(imgdir, exist_ok=True)
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        # print("**configs", configs)

        cli = OmegaConf.from_dotlist(unknown)
        # print("**cli", cli) # cli should be NOne

        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())


        # print("***lc", lightning_config)
        # {'find_unused_parameters': False, 'metrics_over_trainsteps_checkpoint': True, 'modelcheckpoint':
        # {'params': {'every_n_train_steps': 500}}, 'callbacks': {'image_logger': {'target': 'main.ImageLogger',
        # 'params': {'batch_frequency': 500, 'max_images': 32, 'increase_log_steps': False, 'log_first_step': True,
        # 'log_images_kwargs': {'use_ema_scope': False, 'inpaint': False, 'plot_progressive_rows': False,
        # 'plot_diffusion_rows': False, 'N': 32, 'unconditional_guidance_scale': 3.0, 'unconditional_guidance_label':
        # ['']}}}}, 'trainer': {'benchmark': True, 'val_check_interval': 5000000, 'num_sanity_val_steps': 0,
        # 'accumulate_grad_batches': 1}}

        # print("***config2", config)
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        # trainer:
        # benchmark: True
        # val_check_interval: 5000000  # really sorry
        # num_sanity_val_steps: 0
        # accumulate_grad_batches: 1

        # default to ddp
        trainer_config["accelerator"] = "ddp"

        print("*** nondefault_trainer_args:", nondefault_trainer_args(opt))
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)

        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            rank_zero_print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config
        print("*** config.model is ", config.model)
        # model
        print("model path = ", opt.base[0])
        print("opt.base all = ", opt.base)
        # model = create_model(opt.base[0]).cpu()
        model = instantiate_from_config(config.model)
        model.cpu()
        print("***model load is done")

        print("\n\n***config model : " ,config.model.base_learning_rate, config.model.sd_locked,config.model.only_mid_control)
        model.learning_rate = config.model.base_learning_rate
        model.sd_locked = config.model.sd_locked
        model.only_mid_control = config.model.only_mid_control

        # # Configs
        # # resume_path0 = './models/control_sd15_ini.ckpt'  # totorial
        # # resume_path1 = './models/control_v11p_sd15_canny.pth'  # conv 1.1
        # resume_path = 'models/control_sd15_canny.pth'  # conv 1
        # # # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
        # model = create_model('models/yuch_v11p_sd15_canny.yaml').cpu()
        # model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
        # # model.load_state_dict(torch.load(resume_path))
        # model.learning_rate = learning_rate
        # model.sd_locked = sd_locked
        # model.only_mid_control = only_mid_control
        # #dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
        # logger = ImageLogger(batch_frequency=logger_freq)
        # # trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
        # trainer = pl.Trainer(accelerator="gpu", devices=gpus, strategy="ddp", precision=32, callbacks=[logger])
        # # Train!
        # trainer.fit(model, dataset)


        if not opt.finetune_from == "":
            # we are finetuning from a ckpt
            rank_zero_print(f"Attempting to load state from {opt.finetune_from}")



            # old_state = torch.load(opt.finetune_from, map_location="cpu")

            # if "state_dict" in old_state:
            #     rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
            #     old_state = old_state["state_dict"]
            #
            # # Check if we need to port weights from 4ch input to 8ch
            # in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
            # new_state = model.state_dict()
            # in_filters_current = new_state["model.diffusion_model.input_blocks.0.0.weight"]
            # in_shape = in_filters_current.shape
            # if in_shape != in_filters_load.shape:
            #     input_keys = [
            #         "model.diffusion_model.input_blocks.0.0.weight",
            #         "model_ema.diffusion_modelinput_blocks00weight",
            #     ]
            #
            #     for input_key in input_keys:
            #         if input_key not in old_state or input_key not in new_state:
            #             continue
            #         input_weight = new_state[input_key]
            #         if input_weight.size() != old_state[input_key].size():
            #             print(f"Manual init: {input_key}")
            #             input_weight.zero_()
            #             input_weight[:, :4, :, :].copy_(old_state[input_key])
            #             old_state[input_key] = torch.nn.parameter.Parameter(input_weight)

            m, u = model.load_state_dict(load_state_dict(opt.finetune_from, location='cpu'), strict=False)
            # m, u = model.load_state_dict(old_state, strict=False)

            print("missing parameters: " , len(m) , "unkown parameters: ", len(u))
            if len(m) > 0:
                rank_zero_print("missing keys:")
                rank_zero_print(m)
            if len(u) > 0:
                rank_zero_print("unexpected keys:")
                rank_zero_print(u)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            rank_zero_print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        rank_zero_print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "debug": opt.debug,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 500,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            rank_zero_print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': 3,
                         'every_n_train_steps': 500,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        print("*** callbacks_cfg", callbacks_cfg)
        callbacks_cfg["checkpoint_callback"]["params"]['save_top_k'] = -1
        # # callbacks_cfg["checkpoint_callback"]["params"]['save_last'] = None
        callbacks_cfg["checkpoint_callback"]["params"]['filename'] = '{epoch}-{step}'
        # # callbacks_cfg["checkpoint_callback"]["params"]['mode'] = 'min'
        callbacks_cfg["checkpoint_callback"]["params"]['monitor'] = 'global_step'
        # del callbacks_cfg["checkpoint_callback"]["params"]['save_top_k']
        # del callbacks_cfg["checkpoint_callback"]["params"]['save_last']
        # del callbacks_cfg["checkpoint_callback"]["params"]['every_n_train_steps']
        # print("**** callbacks_cfg", callbacks_cfg)
        # from datetime import timedelta
        # delta = timedelta(
        #     minutes=1,
        # )

        # val/loss_simple_ema
        logger = ImageLogger2(batch_frequency=500, log_dir=imgdir)
        trainer_kwargs["callbacks"] = [logger] + [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        # personalization:
        # trainer_kwargs["callbacks"][-1].CHECKPOINT_NAME_LAST = "{epoch}-{step}--last"

        print("plugins in trainer_kwargs? " , "plugins" in trainer_kwargs)
        if not "plugins" in trainer_kwargs:
            trainer_kwargs["plugins"] = list()

        # print("not lightning_config.get : ", not lightning_config.get("find_unused_parameters", True))
        if not lightning_config.get("find_unused_parameters", True):
            print("not lightning_config.get : ", not lightning_config.get("find_unused_parameters", True))
            from pytorch_lightning.plugins import DDPPlugin

            trainer_kwargs["plugins"].append(DDPPlugin(find_unused_parameters=True))

        from pytorch_lightning.plugins import DDPPlugin
        # save ckpt every n steps:
        # checkpoint_callback2 = ModelCheckpoint( monitor='global_step',save_last=True,filename='*cb2{epoch}-{step}', every_n_train_steps=5)
        # trainer_kwargs["callbacks"].append(checkpoint_callback2)

        # logger = ImageLogger(batch_frequency=10, log_dir=imgdir)

        # from pytorch_lightning.callbacks import ModelCheckpoint

        # print("***ckpt dir is :" , ckptdir)
        # checkpoint_callback = ModelCheckpoint(monitor = 'global_step',dirpath = ckptdir,
        #                                       filename = 'control_{epoch}-{step}',verbose=True,
        #                                       every_n_train_steps=10, save_top_k=-1, save_last=True)


        # trainer_kwargs["callbacks"] = [logger, checkpoint_callback]
        print("*** trainer opt " , trainer_opt)
        print("*** trainer kwargs " , trainer_kwargs)
        # gpus = '0,'
        # gpus = '0,1,2,3,4,5,6,7'
        # gpus = getattr(trainer_opt, 'gpus')
        # print("gpus is ", gpus )
        # trainer = pl.Trainer(accelerator="ddp", gpus = gpus, precision=32, callbacks=trainer_kwargs["callbacks"],logger=trainer_kwargs["logger"])
        # trainer = Trainer.from_argparse_args(trainer_opt)
        print("*** log dir is " , logdir)
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # trainer = Trainer(plugins=[DDPPlugin(find_unused_parameters=False)] , accelerator='ddp',
        #                   accumulate_grad_batches=1, benchmark=True, gpus='0,', num_sanity_val_steps=0, val_check_interval=5000000 )
        # # setting for training
        # batch_size = 20
        # root_dir = '/yuch_ws/views_release'
        # num_workers = 16
        # total_view = 12
        logger_freq = 500

        print('*** config.data is ', config.data )

        total_view = config.data['params']['total_view']
        num_workers = config.data['params']['num_workers']
        batch_size = config.data['params']['batch_size']
        root_dir = config.data['params']['root_dir']
        valid_path = config.data['params']['valid_path']
        img_size = config.data['params']['image_size']

        data = ObjaverseDataModuleFromConfig(root_dir, batch_size, total_view, num_workers,valid_path,img_size)
        data.prepare_data()

        data.setup()

        # data = instantiate_from_config(config.data)
        # data.prepare_data()
        # data.setup()
        rank_zero_print("#### Data ####")
        try:
            for k in data.datasets:
                rank_zero_print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
        except:
            rank_zero_print("datasets not yet initialized.")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
            # ngpu = lightning_config.trainer.gpus
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            rank_zero_print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            rank_zero_print("++++ NOT USING LR SCALING ++++")
            rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                rank_zero_print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                if not opt.debug:
                    melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except RuntimeError as err:
        raise err
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            rank_zero_print(trainer.profiler.summary())








