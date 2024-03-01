import json
import sys
import subprocess
import os

file = 'BLIP2_split_by_count_recheck_tag_V5.json'

with open (file,'r') as f:
    x = json.load(f)



for i in range(6,14):
    print(i, len(x[str(i)]))
    to_update = 'valid_paths_' + str(i) + '.json'
    if i != 13:
        to_add = x[str(i)]
        with open(to_update,'r')as f:
            target = json.load(f)
        target = to_add + target

        if i >= 6:
            for id in to_add:
                target_dir = '/yuch_ws/views_release/' + id
                cmd1 = 'mv ' + target_dir +'/BLIP_best_text_v2.txt '+ target_dir+ '/BLIP_best_text_v2_old.txt'
                cmd2 = 'mv ' + target_dir +'/BLIP_best_text_v3.txt '+ target_dir+ '/BLIP_best_text_v2.txt'
                print(cmd1)
                print(cmd2)
                os.system(cmd1)
                os.system(cmd2)

    elif i == 13:
        target = x[str(i)]

    with open(to_update,'w') as f:
        json.dump(target,f)



