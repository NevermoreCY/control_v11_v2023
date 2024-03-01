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
    for job_num in range(0,14):
        # job_num = args.job_num
        tag = 'cat'


        imgs_folder = "/yuch_ws/views_release"

        with open('BLIP2_split_by_count_V4.json') as f:
            data_dict = json.load(f)

        print("processing data for count ", job_num)
        cur_data = data_dict[str(job_num)]
        cur_out_dir = 'data_checking/' + tag + '/' + str(job_num)
        os.makedirs(cur_out_dir,exist_ok=True)
        data_list = []



        for i in tqdm(range(len(cur_data))):
            folder = cur_data[i]
            img_folder = imgs_folder + "/" + folder
            text_path = img_folder + "/BLIP_best_text_v2.txt"
            with open(text_path,'r') as f:
                line = f.readline()
            words = line.split()
            if tag in words or (tag + 's') in words:
                count +=1
                data_list.append(folder)
                for idx in [0,3,7,11]:
                    im_path = os.path.join(img_folder , '%03d.png' % idx)
                    target_path = cur_out_dir + '/' + folder + ('%03d.png' % idx)
                    shutil.copy(im_path,target_path)
                # print(count, line)
            if i % 2000 ==0:
                print(count)

        with open('data_checking/' + tag + '/' + str(job_num) +'.json', 'w') as f :
            json.dump(data_list,f)


if __name__ == '__main__':
    #sys.argv = ["programName.py","--input","test.txt","--output","tmp/test.txt"]
    main()





