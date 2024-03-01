import json
from tqdm import tqdm
import os
# total_jobs= 38
#
# count_dict= {}
# for i in range(14):
#     count_dict[str(i)] = []
#
# for i in range(total_jobs):
#     file_name = "BLIP_v2_data_split_" + str(i) + ".json"
#     with open(file_name, 'r') as f:
#         cur_json = json.load(f)
#     for key in cur_json:
#         # print(type(key))
#         count_dict[key].extend( cur_json[key] )
#         print(i, key, len(cur_json[key]))
#
# for i in range(14):
#     print("for count ", i ," we have ", len( count_dict[str(i)]) , ' samples ')
#
# with open("BLIP2_split_by_count.json", 'w') as f:
#     json.dump(count_dict, f)
#
#--function 2 --------------------------------------------------------
# valid_path = "valid_paths.json"
#
# out = {}
# for i in range(14):
#     out[i] = []
#
# out[-1] = []
# with open(valid_path, 'r') as f:
#     valid = json.load(f)
#
# prefix = "/yuch_ws/views_release/"
# for i in tqdm(range(len(valid))):
#     folder = valid[i]
#     json_path = prefix + folder + "/objarverse_BLIP_metadata_v2.json"
#     if os.path.isfile(json_path):
#         with open(json_path, 'r') as f:
#             meta = json.load(f)
#         out[meta['count']].append(folder)
#     else:
#         out[-1].append(folder)
#
# for i in range(14):
#     print("for count ", i ," we have ", len( out[i]) , ' samples ')
#
# with open("BLIP2_split_by_count.json", 'w') as f:
#     json.dump(out, f)

# ----------------------fuction 3 ------------------------------------------
valid_path = "BLIP2_split_by_count.json"
with open(valid_path, 'r') as f:
    valid = json.load(f)

total = len(valid["-1"])
for i in range(14):
    total += len(valid[str(i)])
print("total is ", total)
print("for count ", -1 ," we have ", len( valid[str(-1)]) , ' samples ' ,len( valid[str(-1)])/total * 100, ' percentage.' )
for i in range(14):
    print("for count ", i ," we have ", len( valid[str(i)]) , ' samples ' ,len( valid[str(i)])/total * 100, ' percentage.' )
#
# for x in range(3,8):
#     out_path = "valid_paths_" + str(x) +".json"
#     valid_x = []
#
#     for i in range(x,14):
#         valid_x.extend(valid[str(i)])
#
#     print(out_path , "total samples are " , len(valid_x), len(valid_x)/ total)
#     with open(out_path, 'w') as f:
#         json.dump(valid_x, f)




