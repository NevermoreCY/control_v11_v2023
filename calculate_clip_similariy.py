import os

from sentence_transformers import SentenceTransformer, util
from PIL import Image


#Load CLIP model
model = SentenceTransformer('clip-ViT-B-32')
models = os.listdir('inference')

for m in models:
    ckpts = os.listdir('inference/' + m )

    for ckpt in ckpts:
        names = os.listdir('inference'+ '/' + m + '/' + ckpt)

        for name in names:
            prompt = name
            images = os.listdir('inference'+ '/' + m + '/' + ckpt + '/' + name)

            for img in images:
                print(prompt)
                clip_dict = {}
                clip_list = []
                img_path = 'inference'+ '/' + m + '/' + ckpt + '/' + name + '/' + img
                img_emb = model.encode(Image.open(img_path))
                text_emb = model.encode([prompt])
                cos_scores = util.cos_sim(img_emb, text_emb)
                cos_value = cos_scores[0][0].item()

                print(cos_scores,cos_value)

                # clip_dict[img] = cos_scores[0][0].value
                # clip_list.append(cos_scores)
                # print(cos_scores.shape)
                # [1,1]









# img_emb = model.encode(Image.open('two_dogs_in_snow.jpg'))
# text_emb = model.encode(['Two dogs in the snow', 'A cat on a table', 'A picture of London at night'])
# cos_scores = util.cos_sim(img_emb, text_emb)

