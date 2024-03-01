import os

import numpy
import torch
from PIL import Image
import time

test_folder = 'test/pure_white/'
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
from lavis.models import load_model_and_preprocess
# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
# prepare the image
# image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

img_names = os.listdir(test_folder)

print("Done preparing")

image_open_t = []
image_process_t = []
q_t= []
for img in img_names:

    print("** For image : ", img)
    print('Single question without remembering previous context:')
    t1 = time.time()
    raw_image = Image.open(test_folder+img).convert("RGB")
    t2 = time.time()-t1
    image_open_t.append(t2)
    print("Image open time: ", t2 , numpy.average(image_open_t) )

    t3 = time.time()
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    t4 = time.time()-t1
    image_process_t.append(t4)
    print("Image process time: ", t4 , numpy.average(image_process_t) )

    print(image.shape, torch.sum(image))
    image = image.repeat(72, 1, 1, 1)

    print(image.shape)

    t5 = time.time()
    q1a= model.generate({"image": image, "prompt": "Question: Can you generate a caption for this image as detail as possible. Including the object's facing direction, color, action and style. This is a object centered png image without background, please ignore the balck background and focus on the object. Also, Don't include word '3D model' in the caption. Answer:"})
    print(q1a)
    t6 = time.time()-t1
    q_t.append(t6)
    print("QA time: ", t6 , numpy.average(q_t) )
    #
    # q1b= model.generate({"image": image, "prompt": "Question: Can you generate a caption for this image as detail as possible. please ignore the black background. Don't use 3d model as keyword.  Answer:"})
    # print(q1b)
    # q2 = model.generate({"image": image, "prompt": "Question: This is a rendering image of a 3D asset, can you tell me whether it is high poly or low poly? Answer:"})
    # print(q2)
    # q3 = model.generate({"image": image, "prompt": "Question: Can you tell me which direction is it facing? Answer:"})
    # print(q3)
    # q4 = model.generate({"image": image, "prompt": "Question: can you tell me what action is this object doing? please ignore the black background Answer:"})
    # print(q4)
    # q5 = model.generate({"image": image, "prompt": "Question: can you tell me the style of this object?  Don't use 3D model as keyword.  Answer:"})
    # print(q5)
    # q6 = model.generate({"image": image, "prompt": "Question: This is a rendering image of a 3D asset, can you tell me whether the object has texture? Answer:"})
    # print(q6)
    # q7 = model.generate({"image": image, "prompt": "Question: Can you tell me whether this is a pure black without any object?  Answer:"})
    # print(q7)
    # q7 = model.generate({"image": image, "prompt": "Question: Can you tell me whether this is a pure white image without any object?  Answer:"})
    # print(q7)
    #
    # print('Ask Question with context:')
    # # cur_prompt = "Question: Can you generate a caption for this image as detail as possible. This is a object centered png image without background, please ignore the balck background and focus on the object. Also, Don't include word '3D model' in the caption.   Answer:"
    # cur_prompt = "Question: This is an object centered image, Can you provide a caption for this object. Ignore the balck or white background. Don't use '3d model' in the caption. Answer:"
    # answer = model.generate({"image": image, "prompt": cur_prompt})
    # print(cur_prompt, answer)
    # Q = 'Can you tell me which direction is it facing?'
    # cur_prompt = 'Question: '+ Q + ' Answer:'
    # answer = model.generate({"image": image, "prompt": cur_prompt})
    # print(Q,answer)
    #
    # Q = 'Can you tell me what action is it doing? Please ignore the black background.'
    # cur_prompt = 'Question: '+ Q + ' Answer:'
    # answer = model.generate({"image": image, "prompt": cur_prompt})
    # print(Q,answer)
    #
    # Q = 'Can you tell me the style of this image? '
    # cur_prompt = 'Question: '+ Q + ' Answer:'
    # answer = model.generate({"image": image, "prompt": cur_prompt})
    # print(Q,answer)
    #
    # Q = 'This is a rendering image of a 3D asset, Can you tell me whether it is high poly or low poly? '
    # cur_prompt = 'Question: '+ Q + ' Answer:'
    # answer = model.generate({"image": image, "prompt": cur_prompt})
    # print(Q,answer)
    #
    # Q = 'Can you tell me whether the object has texture or not? '
    # cur_prompt = 'Question: '+ Q + ' Answer:'
    # answer = model.generate({"image": image, "prompt": cur_prompt})
    # print(Q,answer)
    #
    # Q = 'Can you tell me whether this is a pure black image without any object? '
    # cur_prompt = 'Question: '+ Q + ' Answer:'
    # answer = model.generate({"image": image, "prompt": cur_prompt})
    # print(Q,answer)
    #
    # Q = 'Can you tell me if this is a pure white image without any object? '
    # cur_prompt = 'Question: '+ Q + ' Answer:'
    # answer = model.generate({"image": image, "prompt": cur_prompt})
    # print(Q,answer)

print("Done inferencing")

