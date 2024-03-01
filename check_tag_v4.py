from PIL import Image, ImageStat
import os
from tqdm import tqdm
import json
import sys
import argparse
# import objaverse
from sentence_transformers import SentenceTransformer, util
import torch

def detect_color_image(file, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True):
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
            return 1  # grayscale
        else:
            return 0  # color
        # print "( MSE=",MSE,")"
    elif len(bands)==1:
        return 1

# def is_black_and_white(img_path, saturation_threshold=30):
#
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     saturation = hsv_img[:,:,1]
#     median_saturation = np.median(saturation)
#
#     return median_saturation < saturation_threshold



def doArgs(argList):
    parser = argparse.ArgumentParser()

    #parser.add_argument('-v', "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument('--job_num',type=int, help="Input file name", required=True)
    # parser.add_argument('--output', action="store", dest="outputFn", type=str, help="Output file name", required=True)

    return parser.parse_args(argList)

def remove_useless_tail(texts):
    out = []
    bad_endings = ['in the dark', 'on a black background', 'in the night sky', 'in the sky', 'in the dark sky',
                   'with a black background']

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
        if row[i] and len(cur_texts[i]) > len(best_text):
            # print( best_text, cur_texts[i])
            best_text = cur_texts[i]
    return best_text, count

def main():

    args = doArgs(sys.argv[1:])
    job_num = args.job_num

    model_clip = SentenceTransformer('clip-ViT-L-14')
    img_folder = "/yuch_ws/views_release"

    valid_path = "BLIP2_split_by_count.json"
    with open(valid_path, 'r') as f:
        valid = json.load(f)

    out_put_data = {}

    i = 13
    print("start section ", i)
    folders = valid[str(i)]
    print(folders[:10])

    for i in range(14):
        out_put_data[str(i)] = []


    for j in tqdm(range(len(folders))):

        folder = folders[j]
        meta_path = img_folder + "/" + folder + "/objarverse_BLIP_metadata_v2.json"
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)

        tags = meta_data['tags']
        texts = remove_useless_tail(meta_data['BLIP_texts'])
        best_text = ''
        count = -1
        for tag in tags:
            for text in texts:
                text_s = text.split()

                # try to contain cases for singular and plural in a naive but fast way
                if tag[-1] == 's':
                    tag_alt = tag[:-1]
                else:
                    tag_alt = tag + 's'

                if (tag in text_s or tag_alt in text_s) and len(text) > len(best_text):
                    best_text = text
                    count = 13
        # case when no tag maching is found:
        if best_text == '':
            text_emb = model_clip.encode(texts)  # 12 x D_embd
            cos_scores = util.cos_sim(text_emb, text_emb)  # 12 x 12
            best_text, count = find_best_text(cos_scores, texts, thresh=0.85)


        meta_data['best_text_v4'] = best_text
        meta_data['text_count_v4'] = count

        out_put_data[str(count)].append(folder)


        out_text_name = img_folder + "/" + folder + "/BLIP_best_text_v2.txt"

        with open(out_text_name, 'w') as f:
            f.write(best_text)

        new_meta_path = img_folder + "/" + folder + "/objarverse_BLIP_metadata_v4.json"
        with open(new_meta_path, 'w') as f:
            json.dump(meta_data, f)


    # s = 1
    out_path = "BLIP2_split_by_count_recheck_tag_V4.json"
    with open(out_path,'w') as f:
        json.dump(out_put_data,f )


    return

if __name__ == '__main__':
    #sys.argv = ["programName.py","--input","test.txt","--output","tmp/test.txt"]
    main()


