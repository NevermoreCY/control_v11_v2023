import cv2
import numpy as np
import json

file_path = 'shapenet_v1.json'
views_folder = "/yuch_ws/views_release"


with open(file_path, 'r') as f:
    folders = json.load(f)

c = 0
total = len(folders)
print(total)
bad_path = []
good_path = []

good_dist = {}
bad_dist = {}

for folder in folders:
    img_path = views_folder + '/' + folder + '/000.png'
    img =cv2.imread(img_path)

    prompt_p = views_folder + '/' + folder + '/BLIP_best_text_v2.txt'
    with open(prompt_p,'r') as f:
        prompt = f.readline()

    if img.sum() == 328200:
        bad_path.append(folder)
        if prompt not in bad_dist:
            bad_dist[prompt] = 1
        else:
            bad_dist[prompt] += 1

        print(c , folder)
        c+=1
    else:
        good_path.append(folder)
        if prompt not in good_dist:
            good_dist[prompt] = 1
        else:
            good_dist[prompt] += 1


out_path = 'shapenet_v1_good.json'
with open (out_path,'w') as f:
    json.dump(good_path,f)

out_path = 'shapenet_v1_good_dist.json'
with open (out_path,'w') as f:
    json.dump(good_dist,f)

out_path = 'shapenet_v1_bad.json'
with open (out_path,'w') as f:
    json.dump(bad_path,f)

out_path = 'shapenet_v1_bad_dist.json'
with open (out_path,'w') as f:
    json.dump(bad_dist,f)

print(c, total, c/total)



