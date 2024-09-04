from model import Mlp, caption_generation
import torch
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import json
import torch.nn as nn
from clip import clip
from PIL import Image

device = 'cpu'  #先用cpu算
dataset_name = 'COCO'

clip_model, preprocess = clip.load("RN50x64", device = device)

captions = []
json_path = "../data/COCO/train.json"  # 标题和图片都有
json_labels = json.load(open(json_path, 'r'))
annotations = json_labels

for annotation in annotations[:566720]:  # 566720 原始数据大小
    captions.append(annotation["caption"])  # 把标题都读到captions

#print(captions[556041])

caption_features = torch.load("../feature/COCO/caption_features.pkl", map_location=torch.device(device))
caption_features_norm = caption_features / caption_features.norm(dim = -1, keepdim = True)

image_path = "../example/COCO_val2014_000000353830.jpg"
ori_image = Image.open(image_path)
image = preprocess(ori_image).unsqueeze(0).to(device)

image_feature = clip_model.encode_image(image)  #图片的特征值
image_feature = image_feature / image_feature.norm(dim = -1, keepdim = True)

similarity = image_feature.float() @ caption_features_norm.float().T

#niber = []
#for k in range(1):
_,max_id = torch.max(similarity, dim=1)
print(captions[max_id.item()]) #打印标题中的max_id

#print(similarity[0][max_id]) #item()
#niber.append(caption_features[max_id.item()].unsqueeze(0))
#similarity[0][max_id.item()] = 0



