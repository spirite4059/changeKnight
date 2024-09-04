import torch
import json
from clip import clip
import numpy as np
from PIL import Image

device = "cuda:0"
clip_model, preprocess = clip.load("RN50x64", device = device)

#把reference的代码独立出来

with torch.no_grad():   #这是使用,不是训练

    with open("./data/COCO/coco_test.txt") as image_names_data:
        image_names = image_names_data.readlines()

    image_features = []
    for image_info in image_names:
        image_file = image_info.split('\n')[0]
        image_id = image_dict[image_file]
        image_path = images_path + image_file
        ori_image = Image.open(image_path)

        image = preprocess(ori_image).unsqueeze(0).to(device)   #加载clip软件
        image_feature = clip_model.encode_image(image)          #将图片编码
        image_features.append(image_feature)                    #拓展到图片的特征文件中

    image_features = torch.cat(image_features)
    torch.save(image_features, "./feature/COCO/image_features.pkl")

