import os
import json
import sys
import argparse
import shutil
from tqdm import tqdm



def main():
    imgs_folder = "/yuch_ws/views_release"
    target_path = 'data_checking/cat_gray/13.json'
    target_dir = 'data_checking/cat_gray_data/'
    os.makedirs(target_dir,exist_ok=True)

    with open(target_path, 'r') as f:
        data_list = json.load(f)

    for i in tqdm(range(len(data_list))):
        folder = data_list[i]
        img_folder = imgs_folder + "/" + folder
        target_folder = target_dir + folder
        shutil.copytree(img_folder,target_folder)



if __name__ == '__main__':
    #sys.argv = ["programName.py","--input","test.txt","--output","tmp/test.txt"]
    main()





