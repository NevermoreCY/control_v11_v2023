from share import *
import config

import einops

import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from PIL import Image




def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold,folders):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        # Image.fromarray(detected_map).save('test_images/' + prompt +".png")

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

        output_imgs = []
        for x_sample in results:
            output_imgs.append(Image.fromarray(x_sample.astype(np.uint8)))
        c = 0
        for img in output_imgs:
            out_path = prefix + "control_net_" + str(c) + '.png'
            img.save(out_path)
            c += 1



    return [255 - detected_map] + results


apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

folders = '/yuch_ws/zero123/objaverse-rendering/views_shape'
# folder_list = os.listdir(folders)
import json

good_path = '/yuch_ws/zero123/zero123/good_samples.json'
with open(good_path, 'r') as f:
    folder_list = json.load(f)

#
from tqdm import tqdm
for i in tqdm(range(0, len(folder_list))):
    print(i)
    folder = folder_list[i]

    img_path = folders + '/' + folder + '/' + 'processed_img.png'
    raw_im = Image.open(img_path)

    input_image = np.array(raw_im)
    print("*** our shape", input_image.shape)
    # ***ourshape(256, 256, 3)

    prompt_path = folders + '/' + folder + '/' + 'BLIP_best_text_v2.txt'
    with open (prompt_path,'r') as f:
        prompt = f.readline()

    prefix = folders + '/' + folder + '/'


    # break

    a_prompt= 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    num_samples = 3
    image_resolution = 512
    ddim_steps = 20
    guess_mode = False
    strength = 1
    scale = 9

    seed = np.random.randint(0,2147483647)
    seed_path = folders + '/' + folder + '/' + 'seed.json'
    with open (seed_path,'w') as f:
        json.dump([seed], f)

    eta = 0
    low_threshold = 100
    high_threshold = 200

    process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold,prefix)
