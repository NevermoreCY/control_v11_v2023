import os
import pandas as pd
import sys

file = '/yuch_ws/3DGEN/cap3d/' + 'cap3d_objarverse_all.csv'
captions = pd.read_csv(file, header=None)

import json
import os
from sentence_transformers import SentenceTransformer, util
from PIL import Image



model = SentenceTransformer('clip-ViT-L-14')

data = {}
imgs_folder = "/yuch_ws/views_release/"

clip_score_cap3d = 0
clip_score_ours = 0


#Encode an image:
# img_emb = model.encode(Image.open('two_dogs_in_snow.jpg'))

#Encode text descriptions
# text_emb = model.encode(['Two dogs in the snow', 'A cat on a table', 'A picture of London at night'])

#Compute cosine similarities

# print(cos_scores)

c = 0

for id in range(len(captions[0])):
    item = captions[0][id]
    target_dir = imgs_folder + item

    if os.path.isdir(target_dir):

        cap3d_text = captions[1][id]

        blip_file = target_dir+ '/BLIP_best_text_v2.txt'
        if os.path.isfile(blip_file):
            with open(blip_file,'r') as f:
                blip_text = f.readline()
        elif os.path.isfile(target_dir + '/BLIP_best_text.txt'):
            blip_file = target_dir + '/BLIP_best_text.txt'
            with open(blip_file,'r') as f:
                blip_text = f.readline()
        else:
            continue

        image_file = target_dir+ '/000.png'

        img_emb = model.encode(Image.open(image_file))
        text_emb = model.encode([cap3d_text, blip_text])
        cos_scores = util.cos_sim(img_emb, text_emb)
        print('cos score', cos_scores)
        clip_score_cap3d += cos_scores[0][0]
        clip_score_ours += cos_scores[0][1]
        c += 1

        print(c, ' cap3d: ',clip_score_cap3d/c, ' Blip2: ',clip_score_ours/c)



















