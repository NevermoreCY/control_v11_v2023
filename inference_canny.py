'''
conda activate zero123
cd stable-diffusion
python gradio_new.py 0
'''
import os
import math
import numpy as np
import plotly.graph_objects as go
import time
import torch
from contextlib import nullcontext
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from torch import autocast
from torchvision import transforms
import cv2
from pytorch_lightning import seed_everything
import random

_SHOW_DESC = True
_SHOW_INTERMEDIATE = False
# _SHOW_INTERMEDIATE = True
_GPU_INDEX = 0
# _GPU_INDEX = 2



def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, control, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z, prompt, seed,extra_prompt):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():

            # in original version ,we get c from image and T
            # c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            a_prompt ='best quality'
            n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'

            if extra_prompt:
                prompt = prompt + ', ' + a_prompt

            c = model.get_learned_conditioning(prompt).tile(n_samples, 1, 1)
            c_img = model.get_learned_image_conditioning(input_im).tile(n_samples, 1, 1)
            c_img = c_img.repeat(1,77,1)
            print("*** c image shape is ", c_img.shape)

            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])

            T_text = T[None, None, :].repeat(n_samples, 77, 1).to(c.device)
            # T = T[:, None, :].repeat(1, 77, 1)
            print("*** T shape is ", T_text.shape, "*** c shape is ", c.shape)
            # 4 77 4   ;  4 77 768
            # T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T_text], dim=-1)

            # print("*** c shape after cat is ", c.shape)
            c = model.cc_projection(c)

            T_img = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c_img = torch.cat([c_img, T_text], dim=-1)
            c_img = model.cc_projection(c_img)

            print("*** T image shape is ", T.shape, "*** c image shape is ", c.shape)

            cond = 'text'
            if cond == 'image':
                c = c_img
            elif cond == 'both':
                c = torch.cat((c,c_img) , dim=1)
                print("using both, size is ", c.shape)



            # print("*** c shape after cc projection is ", c.shape)
            # now we get c from text + T

            # print("scale is ", scale)
            # print("*** control shape is ,", type(control ), control.shape)
            # control_encode = model.encode_first_stage(control)
            # print("&&& encode first state", type(control_encode), control_encode.shape)
            # x = control_encode.mode()
            # print("*** control _encode mode is : " , type(x), x.shape)
            # control_encode = model.get_first_stage_encoding(control_encode).detach().repeat(n_samples, 1, 1, 1)
            # use get first stage encoding will result pure noise in output , but we used it in training.
            # print("&&& control_encode shape is ", control_encode.shape)
            # import einops
            # control = einops.rearrange(control, 'h w c -> c h w')
            control = control.detach().repeat(n_samples, 1, 1, 1)
            print("*** control shape is ", control.shape)

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [control]
            print("c2")
            # cond['c_concat'] = [model.encode_first_stage((control.to(c.device))).mode().detach().repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}

                # uc['c_concat'] = [torch.zeros(n_samples, 3, h , w ).to(c.device)]
                # uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]

                if extra_prompt:
                    uc['c_concat'] = [control]
                    uc['c_crossattn'] = [model.get_learned_conditioning(n_prompt).tile(n_samples, 1, 1)]
                else:
                    uc['c_concat'] = [torch.zeros(n_samples, 3, h , w ).to(c.device)]
                    uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]

                #
                # uc_cross = model.get_learned_conditioning([""] * n_samples)
                # print(uc['c_crossattn'][0].shape, uc_cross.shape)
                #
                # uc['c_crossattn'] = [uc_cross]
                print(uc['c_crossattn'][0].shape, uc['c_concat'][0].shape)


                # model.get_learned_conditioning([""] * N)
            else:
                uc = None

            # return 222
            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
# t = 0
#
# if unconditional_guidance_scale > 1.0:
#     uc = self.get_unconditional_conditioning(N, unconditional_guidance_label)
#     if self.model.conditioning_key == "crossattn-adm":
#         uc = {"c_crossattn": [uc], "c_adm": c["c_adm"]}
#     with ema_scope("Sampling with classifier-free guidance"):
#         samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
#                                          ddim_steps=ddim_steps, eta=ddim_eta,
#                                          unconditional_guidance_scale=unconditional_guidance_scale,
#                                          unconditional_conditioning=uc,
#                                          )
#         x_samples_cfg = self.decode_first_stage(samples_cfg)
#         log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
    if ddim:
        ddim_sampler = DDIMSampler(self)
        shape = (self.channels, self.image_size, self.image_size)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                     shape, cond, verbose=False, **kwargs)

    else:
        samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                             return_intermediates=True, **kwargs)

    return samples, intermediates

class CameraVisualizer:
    def __init__(self, gradio_plot):
        self._gradio_plot = gradio_plot
        self._fig = None
        self._polar = 0.0
        self._azimuth = 0.0
        self._radius = 0.0
        self._raw_image = None
        self._8bit_image = None
        self._image_colorscale = None

    def polar_change(self, value):
        self._polar = value
        # return self.update_figure()

    def azimuth_change(self, value):
        self._azimuth = value
        # return self.update_figure()

    def radius_change(self, value):
        self._radius = value
        # return self.update_figure()

    def encode_image(self, raw_image):
        '''
        :param raw_image (H, W, 3) array of uint8 in [0, 255].
        '''
        # https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot

        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        self._raw_image = raw_image
        self._8bit_image = Image.fromarray(raw_image).convert('P', palette='WEB', dither=None)
        # self._8bit_image = Image.fromarray(raw_image.clip(0, 254)).convert(
        #     'P', palette='WEB', dither=None)
        self._image_colorscale = [
            [i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]

        # return self.update_figure()

    def update_figure(self):
        fig = go.Figure()

        if self._raw_image is not None:
            (H, W, C) = self._raw_image.shape

            x = np.zeros((H, W))
            (y, z) = np.meshgrid(np.linspace(-1.0, 1.0, W), np.linspace(1.0, -1.0, H) * H / W)
            print('x:', lo(x))
            print('y:', lo(y))
            print('z:', lo(z))

            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=self._8bit_image,
                cmin=0,
                cmax=255,
                colorscale=self._image_colorscale,
                showscale=False,
                lighting_diffuse=1.0,
                lighting_ambient=1.0,
                lighting_fresnel=1.0,
                lighting_roughness=1.0,
                lighting_specular=0.3))

            scene_bounds = 3.5
            base_radius = 2.5
            zoom_scale = 1.5  # Note that input radius offset is in [-0.5, 0.5].
            fov_deg = 50.0
            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

            input_cone = calc_cam_cone_pts_3d(
                0.0, 0.0, base_radius, fov_deg)  # (5, 3).
            output_cone = calc_cam_cone_pts_3d(
                self._polar, self._azimuth, base_radius + self._radius * zoom_scale, fov_deg)  # (5, 3).
            # print('input_cone:', lo(input_cone).v)
            # print('output_cone:', lo(output_cone).v)

            for (cone, clr, legend) in [(input_cone, 'green', 'Input view'),
                                        (output_cone, 'blue', 'Target view')]:

                for (i, edge) in enumerate(edges):
                    (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                    (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                    (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                    fig.add_trace(go.Scatter3d(
                        x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                        line=dict(color=clr, width=3),
                        name=legend, showlegend=(i == 0)))
                    # text=(legend if i == 0 else None),
                    # textposition='bottom center'))
                    # hoverinfo='text',
                    # hovertext='hovertext'))

                # Add label.
                if cone[0, 2] <= base_radius / 2.0:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] - 0.05], showlegend=False,
                        mode='text', text=legend, textposition='bottom center'))
                else:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] + 0.05], showlegend=False,
                        mode='text', text=legend, textposition='top center'))

            # look at center of scene
            fig.update_layout(
                # width=640,
                # height=480,
                # height=400,
                height=360,
                autosize=True,
                hovermode=False,
                margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                showlegend=True,
                legend=dict(
                    yanchor='bottom',
                    y=0.01,
                    xanchor='right',
                    x=0.99,
                ),
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=1.0),
                    camera=dict(
                        eye=dict(x=base_radius - 1.6, y=0.0, z=0.6),
                        center=dict(x=0.0, y=0.0, z=0.0),
                        up=dict(x=0.0, y=0.0, z=1.0)),
                    xaxis_title='',
                    yaxis_title='',
                    zaxis_title='',
                    xaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    yaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    zaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks='')))

        self._fig = fig
        return fig


def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    print('new input_im:', lo(input_im))

    return input_im


def random_canny(image):
    # compute the median of the single channel pixel intensities
    # apply automatic Canny edge detection using the computed media

    lower = 10 + np.random.random() * 90  # lower 0 ~ 100
    upper = 150 + np.random.random() * 100  # upper 150 ~ 250

    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def fixed_canny(image):
    # compute the median of the single channel pixel intensities
    # apply automatic Canny edge detection using the computed media

    lower = 100  # lower 0 ~ 100
    upper = 200  # upper 150 ~ 250

    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def calc_cam_cone_pts_3d(polar_deg, azimuth_deg, radius_m, fov_deg):
    '''
    :param polar_deg (float).
    :param azimuth_deg (float).
    :param radius_m (float).
    :param fov_deg (float).
    :return (5, 3) array of float with (x, y, z).
    '''
    polar_rad = np.deg2rad(polar_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    fov_rad = np.deg2rad(fov_deg)
    polar_rad = -polar_rad  # NOTE: Inverse of how used_x relates to x.

    # Camera pose center:
    cam_x = radius_m * np.cos(azimuth_rad) * np.cos(polar_rad)
    cam_y = radius_m * np.sin(azimuth_rad) * np.cos(polar_rad)
    cam_z = radius_m * np.sin(polar_rad)

    # Obtain four corners of camera frustum, assuming it is looking at origin.
    # First, obtain camera extrinsics (rotation matrix only):
    camera_R = np.array([[np.cos(azimuth_rad) * np.cos(polar_rad),
                          -np.sin(azimuth_rad),
                          -np.cos(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(azimuth_rad) * np.cos(polar_rad),
                          np.cos(azimuth_rad),
                          -np.sin(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(polar_rad),
                          0.0,
                          np.cos(polar_rad)]])
    # print('camera_R:', lo(camera_R).v)

    # Multiply by corners in camera space to obtain go to space:
    corn1 = [-1.0, np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn2 = [-1.0, -np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn3 = [-1.0, -np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn4 = [-1.0, np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn1 = np.dot(camera_R, corn1)
    corn2 = np.dot(camera_R, corn2)
    corn3 = np.dot(camera_R, corn3)
    corn4 = np.dot(camera_R, corn4)

    # Now attach as offset to actual 3D camera position:
    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2)
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2)
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2)
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1]
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2)
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4]

    return np.array([xs, ys, zs]).T



# def main_run(raw_im, models, device, view, view_name, image_path, prompt, img_size,
#              scale=3.0, n_samples=4, ddim_steps=75, ddim_eta=1.0,
#              precision='fp32'):
#     '''
#     :param raw_im (PIL Image).
#     '''
#     x,y,z = view
#     h = img_size
#     w = img_size
#
#
#     raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
#     input_im = preprocess_image(models, raw_im, True)
#
#     print("*** input_im after process", type(input_im), input_im.shape, np.max(input_im), np.min(input_im))
#
#     # canny edge process
#     show_in_im1 = (input_im * 255.0).astype(np.uint8)
#     print("***show_in_im1 type : ", type(show_in_im1), show_in_im1.shape, np.max(show_in_im1), np.min(show_in_im1))
#     show_in_im2 = Image.fromarray(show_in_im1)
#
#     canny_r = random_canny(show_in_im1)
#     canny_r = canny_r[:, :, None]
#     canny_r = np.concatenate([canny_r, canny_r, canny_r], axis=2)
#     # canny_r = canny_r.astype(np.float32)
#     input_canny = canny_r.astype(np.float32) / 255.0
#
#     print("*** canny after process", type(canny_r), canny_r.shape, np.max(canny_r), np.min(canny_r))
#
#
#     # elif 'gen' in return_what:
#     s = 2
#
#     input_im = transforms.ToTensor()(input_canny).unsqueeze(0).to(device)
#     # input_im = input_im * 2 - 1
#     input_im = transforms.functional.resize(input_im, [h, w])
#
#     # input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
#     # input_im = input_im * 2 - 1
#     # input_im = transforms.functional.resize(input_im, [h, w])
#     print("*** test input im shape is ", type(input_im), input_im.shape)
#
#     sampler = DDIMSampler(models['turncam'])
#     # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
#     used_x = x  # NOTE: Set this way for consistency.
#     x_samples_ddim = sample_model(input_im, models['turncam'], sampler, precision, h, w,
#                                   ddim_steps, n_samples, scale, ddim_eta, used_x, y, z, prompt)
#
#     output_ims = []
#     for x_sample in x_samples_ddim:
#         x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
#         output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
#     c = 0
#     for img in output_ims:
#         out_path = image_path + str(c) + '.png'
#         img.save(out_path)
#         c += 1


def main_run_control(raw_image, models, device, view, view_name, image_path, prompt, img_size,
             scale=3.0, n_samples=3, ddim_steps=75, ddim_eta=1.0,
             precision='fp32',seed=-1,extra_prompt=True):
    '''
    :param raw_im (PIL Image).
    '''
    x,y,z = view
    h = img_size
    w = img_size


    raw_image.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    input_image = preprocess_image(models, raw_image, True)

    # print("*** input_im after process", type(input_im), input_im.shape, np.max(input_im), np.min(input_im))
    # 256 256 3

    # canny edge process
    show_in_im1 = (input_image * 255.0).astype(np.uint8)
    #print("***show_in_im1 type : ", type(show_in_im1), show_in_im1.shape, np.max(show_in_im1), np.min(show_in_im1))
    #256 256 3

    show_in_im2 = Image.fromarray(show_in_im1)

    canny_r = fixed_canny(show_in_im1)
    canny_r = canny_r[:, :, None]
    canny_r = np.concatenate([canny_r, canny_r, canny_r], axis=2)

    if view_name == 'front':
        print("save canny")
        canny = Image.fromarray(canny_r.astype(np.uint8))
        out_path = image_path + 'canny' + '.png'
        canny.save(out_path)

    # canny_r = canny_r.astype(np.float32)
    input_canny = canny_r.astype(np.float32) / 255.0

    # if view == 'front':
    #     canny = Image.fromarray(canny_r.astype(np.uint8))
    #     out_path = image_path + 'canny' + '.png'
    #     canny.save(out_path)
    #print("*** input_canny", type(canny_r), canny_r.shape, np.max(canny_r), np.min(canny_r))
    # 256 256 3

    # elif 'gen' in return_what:
    s = 2

    # print(canny_r.shape)
    input_control = transforms.ToTensor()(input_canny).unsqueeze(0).to(device)
    # # input_im = input_im * 2 - 1
    input_control = transforms.functional.resize(input_control, [h, w])

    print("*** input_im ", type(input_control), input_control.shape)
    # 256 256 3

    # input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    # input_im = input_im * 2 - 1
    # input_im = transforms.functional.resize(input_im, [h, w])
    # print("*** test input im shape is ", type(input_im), input_im.shape)
    # 256 256 3
    input_im = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    input_im = transforms.functional.resize(input_im, [h, w])

    sampler = DDIMSampler(models['turncam'])
    # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
    used_x = x  # NOTE: Set this way for consistency.
    uc_samples_ddim = sample_model(input_im,input_control, models['turncam'], sampler, precision, h, w,
                                  ddim_steps, n_samples, scale, ddim_eta, used_x, y, z, prompt,seed, extra_prompt)


    output_ims = []
    for x_sample in uc_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    c = 0
    for img in output_ims:
        out_path = image_path + str(c) + '.png'
        img.save(out_path)
        c += 1

    # c_samples_ddim = sample_model(input_im, models['turncam'], sampler, precision, h, w,
    #                                ddim_steps, 3, 1, ddim_eta, used_x, y, z, prompt)
    #
    # output_ims = []
    # for x_sample in c_samples_ddim:
    #     x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
    #     output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    # c = 0
    # for img in output_ims:
    #     out_path = image_path  + str(c) + '.png'
    #     img.save(out_path)
    #     c += 1




    # save all the images


    # if 'angles' in return_what:
    #     return (x, y, z, description, new_fig, show_in_im2, output_ims)
    # else:
    #     return (description, new_fig, show_in_im2, output_ims)


def main():

    model_dir = 'control_zeroxl'
    config = 'models/yuch_v11p_zeroxl_canny.yaml'

    # model_dir = 'control_origin_256'
    # config = 'models/yuch_v11p_sd15_origin.yaml'

    config = OmegaConf.load(config)
    # img_size = 256
    img_size = 256
    seed= 666
    use_extra_prompt = False

    device_idx = 0
    device = f'cuda:{device_idx}'


    front = [0, 0, 0]
    left = [0.0, -90.0, 0.0]
    above = [-90.0, 0.0, 0.0]
    right = [0.0, 90.0, 0.0]
    below = [90, 0, 0]
    behind = [0, 180, 0]
    view_list = [front, left, above, right, below, behind]
    view_names = ['front', 'left', 'above', 'right', 'below', 'behind']


    model_list = os.listdir(model_dir)
    model_list.sort()

    images = 'test/shapenet_cases'
    image_list = os.listdir(images)

    for model_name in model_list:
        ckpt_name = model_name.split('.')[0]
        # ckpt = 'ckpt/ckpt_10_16/last.ckpt'
        ckpt = model_dir + "/" + model_name


        # Instantiate all models beforehand for efficiency.
        models = dict()
        print('Instantiating LatentDiffusion... \n')
        print('Instantiating LatentDiffusion... config is ', config)
        print('Instantiating LatentDiffusion... ckpt is ', ckpt)
        models['turncam'] = load_model_from_config(config, ckpt, device=device)
        print('Instantiating Carvekit HiInterface...')
        models['carvekit'] = create_carvekit_interface()
        print('Instantiating StableDiffusionSafetyChecker...')


        for image in image_list:
            print("Inferencing " , image)
            image_name = image.split('.')[0]
            image_dir = 'inference/' + model_dir + '/' +ckpt_name + '/' + image_name
            os.makedirs(image_dir,exist_ok=True)

            prompt = image_name
            raw_image = Image.open(images+ '/' + image)
            # raw_image = 1

            for view_id in range(len(view_list)):
                view_name = view_names[view_id]
                view = view_list[view_id]
                image_path = image_dir + "/" + view_name
                main_run_control(raw_image, models, device, view, view_name, image_path, prompt, img_size,seed=seed,extra_prompt=use_extra_prompt)



if __name__ == '__main__':
    main()