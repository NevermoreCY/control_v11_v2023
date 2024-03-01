import os

import pandas as pd

file = '/yuch_ws/3DGEN/cap3d/' + 'cap3d_objarverse_hq.csv'
captions = pd.read_csv(file, header=None)

import json
import os

target = 'BLIP2_split_by_count_V4.json'

with open(target, 'r') as f:
    valid = json.load(f)



data = {}

cap3_data = {}
cap3_data[3] = []
for i in range(0,14):

    data[i] = valid[str(i)]
    print('data ', i, len(data[i] ))
    cap3_data[i] = []

c = 0
for item in captions[0]:
    c+=1
    # print(c)
    not_found = True
    for i in range(0,14):
        if item in data[i]:
            cap3_data[i].append(item)
            not_found=  False
            break

    if not_found:
        cap3_data[3].append(item)

    if c %10000 == 0:
        for i in range(0, 14):
            print(c, i, len(cap3_data[i]))


for i in range(0,14):
    print(i, len(cap3_data[i]))
out_path = 'cap3d_distribution.json'
with open(out_path,'w') as f:
    json.dump(out_path)







## if u want to obtain the caption for specific UID

#
#
#
# for item in captions:
#     print(item)
#     # print('cap3d:',captions[captions[0] == item][1].values[0])
#     data_path = '/home/nev/3DGEN/Results/12_11/cap3D/to_check/' + item + '/BLIP_best_text_v2.txt'
#     with open(data_path,'r')as f:
#         text = f.readline()
#     cap3d = captions[captions[0] == item][1].values[0]
#     print('cap3d:', cap3d)
#     print('Our cap:', text)
#
#     out_cap3d = '/home/nev/3DGEN/Results/12_11/cap3D/to_check/' + item + '/cap3d_' + cap3d
#     with open(out_cap3d,'w') as f:
#         f.write(cap3d)
#     out_our = '/home/nev/3DGEN/Results/12_11/cap3D/to_check/' + item + '/blip2_' + text
#     with open(out_our,'w') as f:
#         f.write(cap3d)
