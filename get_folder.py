import json

with open('valid_paths.json') as f:
    x = json.load(f)

x.sort()

