import json
from tqdm import tqdm

for i in range(6,13):
    print("job ", i)
    animal_gray_path = 'animal_grayscale_data_list.json'
    target_path = 'valid_paths_' + str(i) + ".json"
    out_path = 'valid_paths_WOGrayAnimal' + str(i) + '.json'

    with open(animal_gray_path,'r') as f:
        to_remove = json.load(f)
    with open(target_path, 'r') as f:
        valid = json.load(f)

    to_save = []

    for i in tqdm(range(len(valid))):
        item = valid[i]
        if item not in to_remove:
            to_save.append(item)

    with open(out_path, 'w') as f:
        json.dump(to_save,f)





