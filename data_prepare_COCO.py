import torch
import json
from clip import clip
import numpy as np
from PIL import Image

device = "cuda:0"
clip_model, preprocess = clip.load("RN50x64", device = device)

with torch.no_grad():   #这是使用,不是训练
    captions = []
    json_path = "./data/COCO/train.json"  #标题和图片都有
    json_labels = json.load(open(json_path,'r'))
    annotations = json_labels
    
    for annotation in annotations[:566720]:             #566720 原始数据大小
        captions.append(annotation["caption"])          #把标题都读到captions
    
    features = []  #256个标题一个特征，56万全部放进去
    index = 0
    batch_size = 256
    while index < len(captions):    #56万个原始标题
        batch_captions = captions[index : index+batch_size]
        clip_captions = clip.tokenize(batch_captions).to(device)    #clip文字编码 ,长度77，向量化文字
        clip_features = clip_model.encode_text(clip_captions)       #转成文字的feature,编码，256个一组进行的编码
        features.append(clip_features)          #所有文字特征  1行2213列
        index += batch_size

    caption_features = torch.cat(features)      #torch.cat是将两个张量（tensor）拼接在一起  打印 x2.shape，应该还是1行2213列？
    torch.save(caption_features, "D:/SCI/Knight-main/feature/COCO/caption_features.pkl")    #编码后的所有标题特征 ，55M
    #torch.save(caption_features, "./feature/COCO/caption_features.pkl")
    captions = np.array(captions)  #
    np.save("./feature/COCO/captions.npy", captions)        #编码前标题
    
    caption_features = caption_features / caption_features.norm(dim = -1, keepdim = True)  #做了一次单一化

    #这里是按照标题的编码找最接近的5个
    nibers = []  #保存相邻标题特征
    print("特征值循环次数：caption_features.shape[0]={caption_features.shape[0]}")
    for i in range(caption_features.shape[0]):              #循环2213次？ 那么就是每256个里面找相似？？
        caption_feature = caption_features[i].unsqueeze(0)  # 拿一组出来； 在指定位置添加一个新维度，并返回新tensor
        similarity = caption_feature @ caption_features.T   #自己和自己找相似度 @是矩阵乘法的重写 matmul
        similarity[0][i] = 0        #？？

        #256个里面找5个
        niber = []
        for j in range(5):  #找最相似的5个值
            _, max_id = torch.max(similarity, dim = 1)      #可用于找到输入张量中所有元素的最大值,dim指定在哪个维度找  similarity维度1是什么？
            niber.append(max_id.item())                     #？max_id.item应该是个数值（也应该是个tensor)
            similarity[0][max_id.item()] = 0                #刚才找到的去掉，最接近的去掉
        nibers.append(niber)                                #从similar属性里面找的

    nibers = np.array(nibers)
    np.save("./feature/COCO/nibers.npy", nibers)        #很小，11M

    json_path = "./data/COCO/captions_val2014.json"         #coco真实的数据数据保存到这里
    json_labels = json.load(open(json_path,'r'))            #找不到这个文件
    annotations = json_labels["annotations"]                #没用上
    images = json_labels["images"]
    images_path = "./data/COCO/image/"

    image_dict = dict()
    for image in images:
        image_dict[image["file_name"]] = image["id"]

#下面是把图片编码，图片是测试集里面的
    # with open("./data/COCO/coco_test.txt") as image_names_data:
    #     image_names = image_names_data.readlines()
    #
    # image_features = []
    # for image_info in image_names:
    #     image_file = image_info.split('\n')[0]
    #     image_id = image_dict[image_file]
    #     image_path = images_path + image_file
    #     ori_image = Image.open(image_path)
    #
    #     image = preprocess(ori_image).unsqueeze(0).to(device)   #加载clip软件
    #     image_feature = clip_model.encode_image(image)          #将图片编码
    #     image_features.append(image_feature)                    #拓展到图片的特征文件中
    #
    # image_features = torch.cat(image_features)
    # torch.save(image_features, "./feature/COCO/image_features.pkl")

