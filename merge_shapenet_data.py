import json
import os


shape_path = 'shapenet_v1_good.json'
turbo_path = 'turbo_v1.json'
turbo_scale = 1
shape_scale = 1


with open(shape_path, 'r') as f:
    shape_data = json.load(f)

with open(turbo_path, 'r') as f:
    turbo_data = json.load(f)

turbo_data = turbo_scale * turbo_data
shape_data = shape_scale * shape_data


print("turbo data final length: " , len(turbo_data))
print("shape data final length: " , len(shape_data))

for i in range(4,14):

    update_path = 'valid_paths_' + str(i) + '.json'
    with open (update_path,'r') as f:
        update_data = json.load(f)

    update_data = shape_data + turbo_data +update_data

    out_path = 'valid_merged_paths_r1_' + str(i) + '.json'
    with open(out_path,'w') as f:
        json.dump(update_data, f)




