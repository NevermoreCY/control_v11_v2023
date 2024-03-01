import json
import os


target = 'BLIP2_split_by_count_V4.json'

with open(target, 'r') as f:
    valid = json.load(f)

for i in range(6,14):
    out_file = 'valid_paths_' + str(i) + '.json'
    out_data = []
    for j in range(i,14):
        out_data.extend(valid[str(j)])

    with open(out_file, 'w') as f:
        json.dump(out_data,f)













