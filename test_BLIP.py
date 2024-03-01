import os
import time
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
# curr = time.time()
# print("time before canny edge", curr)
# img = cv2.imread(prefix + file_id + ".png")
# next_t = time.time()
# print(" time after canny edge =", next_t)
# canny_a = auto_canny(img)
# print("diff 1", next_t - curr)
from torchvision.io import read_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is ", device )

cwd = os.getcwd()
print("cwd is ", cwd)
os.chdir("BLIP")
cwd = os.getcwd()
print("cwd is ", cwd)
def load_demo_image(image_size, device):
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    w, h = raw_image.size
    #display(raw_image.resize((w // 5, h // 5)))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image
def load_image(image_size, device, im_path):

    raw_image = Image.open(im_path).convert('RGB')
    print("raw_image" , raw_image.size)
    image_a = np.array(raw_image)
    print(image_a.shape)
    image_a = image_a.transpose(2,0,1)
    print(image_a.shape)

    image_stack = np.stack([image_a, image_a,image_a],axis=0)
    print(image_stack.shape)

    raw_images = [raw_image,raw_image]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    # image = transform(raw_image).unsqueeze(0).to(device)
    image = transform(raw_image).to(device)
    print("**image shape", image.shape)
    return image


def load_image2(image_size, device, im_path):

    raw_image = Image.open(im_path).convert('RGB')
    print("raw_image" , raw_image.size)

    transform1 = transforms.Compose([
        transforms.ToTensor()
    ])

    raw_image = transform1(raw_image).to(device)
    # raw_images = [raw_image,raw_image]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    # image = transform(raw_image).unsqueeze(0).to(device)
    image = transform(raw_image)
    print("**image shape", image.shape)
    return image


from BLIP.models.blip import blip_decoder






model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
image_size = 512

# model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
# model.eval()
# model = model.to(device)


cwd = os.getcwd()
print("cwd is ", cwd)
os.chdir("../")
cwd = os.getcwd()
print("cwd is ", cwd)

img_folder = "objvarse_views"
sub_folder_list = os.listdir(img_folder)
sub_folder_list.sort()

def most_frequent(List):
    counter = 0
    item = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            item = i
    return item

n= 0
for folder in sub_folder_list:
    n+=1
    if folder[-4:] != "json":
        texts = []
        for i in range(0):
            im_path = os.path.join(img_folder + "/" + folder, '%03d.png' % i)
            print(im_path)

            curr = time.time()
            print("time load_image", curr)
            x = load_image(image_size=image_size, device=device, im_path=im_path)
            y = load_image2(image_size=image_size, device=device, im_path=im_path)

            print(x == y)

            next_t = time.time()
            print(" time after load_image =", next_t)
            print("diff 1", next_t - curr)

            print("x shape is ", x.shape)

            image = torch.stack([x,x,x], 0)
            print(image.shape)



            with torch.no_grad():
                # beam search
                curr = time.time()
                print("time inference", curr)
                caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
                print("caption shape ", caption.shape)
                next_t = time.time()
                print(" time after inference =", next_t)
                print("diff 2", next_t - curr)
                # nucleus sampling
                # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        #        print('caption: ' + caption[0])
                texts.append(caption[0])
        out_text_name = img_folder + "/" + folder + "/BLIP_best_text.txt"
        print(n,out_text_name , texts)
        # name = most_frequent(texts)
        name = "test prompt ! "
        with open(out_text_name, 'w') as f:
            f.write(name)
        print(n)

