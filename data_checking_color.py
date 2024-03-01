import os
import json
import sys
import argparse
import shutil
from tqdm import tqdm

def doArgs(argList):
    parser = argparse.ArgumentParser()

    #parser.add_argument('-v', "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument('--job_num',type=int, help="Input file name", required=True)
    # parser.add_argument('--output', action="store", dest="outputFn", type=str, help="Output file name", required=True)

    return parser.parse_args(argList)


def main():
    args = doArgs(sys.argv[1:])
    count = 0

    animal_data_colored = []
    animal_data_grayscale = []

    for job_num in range(6,14):
        # job_num = args.job_num
        target_tags = ['cat', 'cats', 'dog', 'dogs', 'animal','bird','fish','creature','man','people']
        categories =  ['animals-pets' , 'characters-creatures']
        imgs_folder = "/yuch_ws/views_release"

        with open('BLIP2_split_by_count_V4.json') as f:
            data_dict = json.load(f)

        print("processing data for count ", job_num)
        cur_data = data_dict[str(job_num)]
        cur_out_dir = 'data_checking/' + 'cat_colored' + '/' + str(job_num)
        os.makedirs(cur_out_dir,exist_ok=True)

        gray_out_dir = 'data_checking/' + 'cat_gray' + '/' + str(job_num)
        os.makedirs(gray_out_dir,exist_ok=True)

        cat_data_colored = []
        cat_data_gray = []



        for i in tqdm(range(len(cur_data))):
            folder = cur_data[i]
            img_folder = imgs_folder + "/" + folder
            data_path = img_folder + "/objarverse_BLIP_metadata_v3.json"
            with open(data_path,'r') as f:
                metadata = json.load(f)

            data_tag = metadata['tags']
            data_cate = metadata['categories']
            data_gray = metadata['grayscale']

            is_animal =False
            is_cat = False
            for tag in target_tags:
                if tag in data_tag:
                    is_animal = True
                    if tag in ['cat','cats']:
                        is_cat = True
                    break
            if not is_animal:
                for cate in categories:
                    if cate in data_cate:
                        is_animal = True



            if is_animal:
                count +=1
                if not data_gray:
                    animal_data_colored.append(folder)
                    if is_cat:
                        cat_data_colored.append(folder)
                        # for idx in [0,3,7,11]:
                        #     im_path = os.path.join(img_folder , '%03d.png' % idx)
                        #     target_path = cur_out_dir + '/' + folder + ('%03d.png' % idx)
                        #     shutil.copy(im_path,target_path)
                    # print(count, line)
                else:
                    animal_data_grayscale.append(folder)
                    if is_cat:
                        cat_data_gray.append(folder)
                        for idx in [0,3,7,11]:
                            im_path = os.path.join(img_folder , '%03d.png' % idx)
                            target_path = gray_out_dir + '/' + folder + ('%03d.png' % idx)
                            shutil.copy(im_path,target_path)
            if i % 2000 ==0:
                print(count)

        # with open('data_checking/' + 'cat_colored' + '/' + str(job_num) +'.json', 'w') as f :
        #     json.dump(cat_data_colored,f)

        with open('data_checking/' + 'cat_gray' + '/' + str(job_num) +'.json', 'w') as f :
            json.dump(cat_data_gray,f)

    # with open('animal_colored_data_list.json', 'w') as f:
    #     json.dump(animal_data_colored,f)
    # with open('animal_grayscale_data_list.json', 'w') as f:
    #     json.dump(animal_data_grayscale, f)



if __name__ == '__main__':
    #sys.argv = ["programName.py","--input","test.txt","--output","tmp/test.txt"]
    main()




