import os

from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import json
import time
import numpy as np
import sys
from torchvision.io import read_image
import argparse

import objaverse
from sentence_transformers import SentenceTransformer, util

import nltk
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
# log = open("image_caption_logs/sep10_job0_t1.log", "a")
# sys.stdout = log
# sys.stderr = log


def doArgs(argList):
    parser = argparse.ArgumentParser()

    #parser.add_argument('-v', "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument('--job_num',type=int, help="Input file name", required=True)
    # parser.add_argument('--output', action="store", dest="outputFn", type=str, help="Output file name", required=True)

    return parser.parse_args(argList)

def main():
    args = doArgs(sys.argv[1:])

    job_num = args.job_num


    # print "Starting %s" % (progName)
    startTime = float(time.time())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cwd = os.getcwd()
    print("cwd is ", cwd)
    os.chdir("BLIP")
    cwd = os.getcwd()
    print("cwd is ", cwd)
    print("device is ", device)

    def load_demo_image(image_size, device):
        img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

        w, h = raw_image.size
        # display(raw_image.resize((w // 5, h // 5)))

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        image = transform(raw_image).unsqueeze(0).to(device)
        return image

    image_size = 256
    transform1 = transforms.Compose([
        transforms.ToTensor()
    ])

    transform2 = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    def load_image(image_size, device, im_path):

        # load_1 = time.time()
        raw_image = Image.open(im_path).convert('RGB')
        # load_2 = time.time()
        # print("load_diff1" , load_2-load_1)
        raw_image = transform1(raw_image).to(device)
        # load_3 = time.time()
        # print("load_diff2", load_3 - load_2)
        #
        # load_5
        # raw_image2 = read_image(im_path).to(device)
        # raw_image2 = raw_image2.type('torch.FloatTensor')
        # load_4 = time.time()
        # print("load_diff3", load_4 - load_3)
        # print(raw_image.shape , raw_image2[:3].shape)

        # print("raw_image shape", raw_image.size)
        # print("raw_image type", type(raw_image))

        # image = transform(raw_image).unsqueeze(0).to(device)
        # image = transform2(raw_image).to(device)
        image = transform2(raw_image).to(device)
        # load_5 = time.time()
        # print("load_diff4", load_5 - load_4)

        return image

    from BLIP.models.blip import blip_decoder

    model_clip = SentenceTransformer('clip-ViT-L-14')


    s = 0
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    image_size = 256

    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    cwd = os.getcwd()
    print("cwd is ", cwd)
    os.chdir("../")
    cwd = os.getcwd()
    print("cwd is ", cwd)

    def most_frequent(List):
        counter = 0
        item = List[0]
        for i in List:
            curr_frequency = List.count(i)
            if (curr_frequency > counter):
                counter = curr_frequency
                item = i
        return item

    def extract_tags(tags):
        out = []
        for item in tags:
            out.append(item['name'])
        return out

    # def extract_category(cate):
    #     out = []
    #     for item in cate:
    #         for dic in item:
    #             out.append(dic['name'])
    #     return out

    def remove_useless_tail(texts):
        out = []
        bad_endings = ['in the dark', 'on a black background', 'in the night sky', 'in the sky','in the dark sky', 'with a black background']

        for text in texts:
            for bad_ending in bad_endings:
                l = len(bad_ending)
                if bad_ending in text and text[-l:] == bad_ending:
                    text = text[:-l]
            out.append(text)
        return out

    def find_best_text(cos_scores, cur_texts, thresh=0.9):

        best_text = ''
        l = len(cur_texts)
        good = cos_scores >= thresh
        idx = torch.argmax(good.sum(dim=1))
        count = torch.max(good.sum(dim=1)).item()

        row = good[idx]
        # print("row", row)
        for i in range(len(row)):
            if row[i]  and len(cur_texts[i]) > len(best_text):
                # print( best_text, cur_texts[i])
                best_text = cur_texts[i]
        return best_text, count






    img_folder = "/yuch_ws/views_release"
    # sub_folder_list = os.listdir(img_folder)
    # sub_folder_list.sort()

    with open('valid_paths.json') as f:
        sub_folder_list = json.load(f)

    sub_folder_list.sort()

    total_n = len(sub_folder_list)
    print("total_n", total_n)  # 772870

    # job_num = 21
    job_length = 20000

    start_n = job_num* job_length
    end_n = (job_num+1) * job_length
    bz = 10

    print("******** cur job_num is ", job_num, "start is", start_n, "end is", end_n)
    print("first few names", sub_folder_list[start_n:start_n + 5])

    # batch_s = start_n
    batch_s = end_n-1500
    batch_e = batch_s + bz

    bad_folders = []

    data_split_by_count = {}
    for i in range(14):
        data_split_by_count[i] = []



    while batch_s < end_n:
        print(batch_s, batch_e)
        iter_time_s = time.time()

        batch_names = sub_folder_list[batch_s:batch_e]

        annotations = objaverse.load_annotations(batch_names)

        might_useful = ['name', 'tags', 'categories', 'description']
        data_dict = {}

        for key in annotations:
            data = annotations[key]
            data_dict[key] = {}
            data_dict[key]['name'] = data['name']
            data_dict[key]['tags'] = extract_tags(data['tags'])
            data_dict[key]['categories'] = extract_tags(data['categories'])
            data_dict[key]['description'] = data['description']
            # data_list[0].append(data['name'])
            # data_list[1].append(extract_tags(data['tags']))
            # data_list[2].append(extract_tags(data['categories']))
            # data_list[3].append(data['description'])
        # print(len(data_list[0]) ,len(data_list[1]),len(data_list[2]),len(data_list[3]) )
        images = []

        curr = time.time()
        # print("time load_image", curr)
        skip_index = []
        target_index = [0, 1,2,3,4,5,6,7, 8,9,10,11]
        views = len(target_index)

        for j in range(bz):
            #print(j)
            folder = batch_names[j]
            if folder[-4:] != "json":
                for idx in range(views):
                    i = target_index[idx]
                    im_path = os.path.join(img_folder + "/" + folder, '%03d.png' % i)
                    if not os.path.isfile(im_path):
                        bad_folders.append(folder)
                        images = images[:-idx]
                        skip_index.append(j)

                        # save the bad items
                        out_text_name = "logs/Bad_folder_names_job_" + str(job_num) + ".txt"
                        with open(out_text_name, 'w') as f:
                            for line in bad_folders:
                                f.write(line + "\n")

                        break

                    images.append(load_image(image_size=image_size, device=device, im_path=im_path))
        # print(assert(bz*12 == len(images)) )
        next_t = time.time()
        # print(" time after load_image =", next_t)
        print("time for load diff 1", next_t - curr)

        print("total num is ", len(images), ", should be", (bz - len(skip_index)) * views)

        # make them a batch
        batch_images = torch.stack(images, 0)
        print("batch shape is ", batch_images.shape)
        with torch.no_grad():
            # beam search
            curr = time.time()
            # print("time before inference", curr)
            captions = model.generate(batch_images, sample=False, num_beams=3, max_length=20, min_length=5)

            next_t = time.time()
            # print(" time after inference =", next_t)
            print("time for inference diff 2", next_t - curr)
            # nucleus sampling
            # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
            #        print('caption: ' + caption[0])

        # post process
        curr = time.time()
        # print("time before post", curr)
        print("num of captions is ", len(captions), "should be ", (bz - len(skip_index)) * views)
        print("skip_index is", skip_index)
        offset = 0
        for j in range(bz):
            if j not in skip_index: # skip if file is not found
                folder = batch_names[j]
                cur_texts = remove_useless_tail(captions[(j + offset) * views:(j + 1 + offset) * views])
                cur_tags = data_dict[folder]['tags']

                best_text = ''
                # rule 1 : if tag is included in the BLIP's text, then it's highly likely to be good text
                # if both sentence contains the tag, we prefer the longer sentence. (we encourage longer description)
                for tag in cur_tags:
                    for text in cur_texts:
                        if tag in text and len(text) > len(best_text):
                            best_text = text
                            count = 13 # 13 means selected by Rule 1
                # case when not tag is found in the texts, we perform a vote
                if best_text == '':
                    text_emb = model_clip.encode(cur_texts) # 12 x D_embd
                    cos_scores = util.cos_sim(text_emb, text_emb) # 12 x 12
                    best_text , count = find_best_text(cos_scores, cur_texts, thresh=0.85)




                # print(len(cur_texts))
                # best_text = most_frequent(cur_texts)
                # out_text_name = img_folder + "/" + folder + "/BLIP_v2_best_text.txt"


                meta_data = data_dict[folder]
                # meta_data["name"] = data_list[0][j]
                # meta_data['tags'] = data_list[1][j]
                # meta_data['categories'] = data_list[2][j]
                # meta_data['description'] = data_list[3][j]
                meta_data["BLIP_texts"] = cur_texts
                meta_data['count'] = count
                meta_data['Best_text'] = best_text

                data_split_by_count[count].append(batch_names[j])

                out_text_name = img_folder + "/" + folder + "/BLIP_best_text_v2.txt"
                out_objarverse_metadata = img_folder + "/" + folder + "/objarverse_BLIP_metadata_v2.json"


                with open(out_text_name, 'w') as f:
                    f.write(best_text)

                with open(out_objarverse_metadata, 'w') as f:
                    json.dump(meta_data, f )
            else:
                offset -= 1
        # print(" time after post =", next_t)
        print("time for post diff 3", next_t - curr)

        # update id
        batch_s += bz
        batch_e += bz
        time_cost = time.time() - iter_time_s
        print("when bz is ", bz, ", 1 iteration takes time :", time_cost / 60, " minutes.", "1 sample will take ",
              time_cost / bz, " seconds")

        print(batch_s, batch_e, end_n)


    data_out = "BLIP_v2_data_split_" + str(job_num) + ".json"
    with open(data_out, 'w') as f:
        json.dump(data_split_by_count,f)



    return

if __name__ == '__main__':
    #sys.argv = ["programName.py","--input","test.txt","--output","tmp/test.txt"]
    main()






