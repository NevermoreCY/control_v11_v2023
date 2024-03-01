import objaverse
print(objaverse.__version__)
import os

obj_anno = objaverse.load_annotations()

#'jT6IwFnaIyIULBQIh2tjp6W4VID'  a person on skateboard

# 'jWqCfOnq5sU58oH0yyGEMN2ddWC' a bug on black background

target_uids = ['jT6IwFnaIyIULBQIh2tjp6W4VID' ,'jWqCfOnq5sU58oH0yyGEMN2ddWC' , '07b85c020dc944f18fae370a7d7b0c14','00366db8bb464e5c8b26703c70a686bc','029e6683aca54614ab97c9596f24aa4d']

target_BLIP_caption = ["a person on skateboard" , "a bug on black background",  "coffee table with cup of coffee"]