import os

from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch

img_folder = 'clip_test_images'
#Load CLIP model
model = SentenceTransformer('clip-ViT-B-32')
names = os.listdir(img_folder)

# candidates = ['an elephant', 'a space ship', 'a train' , 'a tall man', 'a air craft' , 'a chair' , 'a house', 'a white cat']

colors = ['yellow','black','white','blue']
facings = ['front', 'side', 'top', 'below']
animals = ['cat', 'dog', 'lion', 'tiger']
candidates = []

for color in colors:
    for facing in facings:
        for animal in animals:
            sent1 = 'a ' + color + ' ' + animal + " , " + facing + " view"
            sent2 = facing + " view of a " + color + ' ' + animal

            candidates.append(sent1)
            candidates.append(sent2)

for name in names:
    prompt = name.split('.')[0]
    print(prompt)
    # clip_dict = {}
    # clip_list = []
    img_path = img_folder + '/' +name
    img_emb = model.encode(Image.open(img_path))
    # text_emb = model.encode([prompt] + candidates)
    text_emb = model.encode( candidates)
    cos_scores = util.cos_sim(img_emb, text_emb)
    args = torch.argsort(cos_scores[0], descending=True)
    # 1 x 128
    # cos_value = cos_scores[0][0].item()
    # print(cos_scores.shape)
    # print("For image : ", name)
    # for i in range(len(candidates))
    # print(name, cos_scores)
    for idx in args:
        print('score : ', cos_scores[0][idx] , ' Sent: ' , candidates[idx])





