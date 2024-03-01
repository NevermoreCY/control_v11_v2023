import json
import cv2
import numpy as np
from PIL import Image, ImageStat
from tqdm import tqdm
import os
import shutil

# 23.5

def detect_color_image(file, thumb_size=256, MSE_cutoff=22, adjust_color_bias=False):
    pil_img = Image.open(file)
    bands = pil_img.getbands()
    if bands == ('R','G','B') or bands== ('R','G','B','A'):
        thumb = pil_img.resize((thumb_size,thumb_size))
        SSE, bias = 0, [0,0,0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias)/3 for b in bias ]
        # print("b" , bias )
        for pixel in thumb.getdata():
            mu = sum(pixel[:3])/3
            # print("p,m",pixel , mu)
            SSE += sum((pixel[i] - mu - bias[i])*(pixel[i] - mu - bias[i]) for i in [0,1,2])
        MSE = float(SSE)/(thumb_size*thumb_size)
        # print("MSE is ", MSE )
        if MSE <= MSE_cutoff:
            return 1 ,MSE   # grayscale
        else:
            return 0 ,MSE  # color
        # print "( MSE=",MSE,")"
    elif len(bands)==1:
        return 1 ,-1

target_dir = '/home/nev/3DGEN/Results/cat data/cat_gray_data/'
folders = os.listdir(target_dir)

x= 0
for folder in folders:
    x+=1
    print(x)
    has_color = False
    img_dir = target_dir + '/' + folder
    meta_data_path = img_dir + '/' +  'objarverse_BLIP_metadata_v3.json'
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    grayscale_MSE = []
    for i in range(12):
        img_path = img_dir +'/' + ('%03d.png' % i)
        c,mse = detect_color_image(img_path)
        grayscale_MSE.append((i,c,mse))
        if c ==0:
            has_color = True

    if has_color:
        save_dir = '/home/nev/3DGEN/Results/cat data/cat_gray_data_colored/'
        target_folder = save_dir + folder
        shutil.copytree(img_dir, target_folder)

    else:
        save_dir = '/home/nev/3DGEN/Results/cat data/cat_gray_data_gray/'
        target_folder = save_dir + folder
        shutil.copytree(img_dir, target_folder)

    meta_data['grayscale_mse_thumb256_no_bias'] = grayscale_MSE


    with open(meta_data_path,'w') as f:
        json.dump(meta_data,f )

#
# t_file = 'test_grayscale/t9.png'
# pil_img = Image.open(t_file)
# a = pil_img
#
# for img in imgs:
#     img_path = 'test_grayscale/' + img
#     print("img is ", img , " detect_color_image : ", detect_color_image(img_path) )
