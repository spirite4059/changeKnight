from model import Mlp, caption_generation
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import json
from tqdm import tqdm
import torch.nn as nn
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_point", type=int, default=468201)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)  #默认32，显存溢出，所有改成16试试
    parser.add_argument("--lr", type=float, default=0.000001)
    parser.add_argument("--epoch", type=int, default=5) 
    parser.add_argument("--weight_decay", type=float, default=0.0000001)
    parser.add_argument("--mlp_lr", type=float, default=10.0)
    parser.add_argument("--prefix", type=str, default="prefix prefix prefix prefix prefix:")
    parser.add_argument("--max_length", type=int, default=40)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--print_frequency", type=int, default=32)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint/")
    parser.add_argument("--output_dir", type=str, default="./output/")
    args = parser.parse_args()
    return args

def main(args):
    device = args.device
    
    if args.dataset == 'coco':
        dataset_name = 'COCO'
    elif args.dataset == 'flickr':
        dataset_name = 'Flickr'
    else:
        print('Please input correct dataset!')
        assertargs.dataset == 'coco' or args.dataset == 'flickr'

    #niber是准备好的标题
    nibers = np.load("./feature/" + dataset_name + "/nibers.npy")    #标题与标题的近似特征值
    nibers = nibers.tolist()

    captions = np.load("./feature/" + dataset_name + "/captions.npy")
    captions = captions.tolist()
    caption_features = torch.load("./feature/" + dataset_name + "/caption_features.pkl").to(device)
    image_features = torch.load("./feature/" + dataset_name + "/image_features.pkl").to(device)
    caption_features_norm = caption_features / caption_features.norm(dim = -1, keepdim = True)
    
    GPT_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")  #没看到输入？
    GPT_tokenizer.pad_token = GPT_tokenizer.eos_token
    GPT_tokenizer.cls_token = GPT_tokenizer.eos_token
    GPT_tokenizer.sep_token = GPT_tokenizer.eos_token
    GPT_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    GPT_model.to(device)

    Map = Mlp()  #多层感知模型
    Map.to(device)
    
    batch_size = args.batch_size
    prefix = args.prefix
                                  #？用GPT的参数，是训练GPT的参数？  用标题特征值和标题文字训练GPT，让它知道近似关系
    optimizer = torch.optim.AdamW([{"params": GPT_model.parameters()}, 
                                   {"params": Map.parameters(), "lr": args.lr*args.mlp_lr}], 
                                  lr = args.lr, weight_decay = args.weight_decay)
    seed = args.seed
    torch.manual_seed(seed)

    # 每一轮训练一次，然后保存，不断逼近模型。训练集不同，就是会有偏离的问题
    for epoch in range(args.epoch):
        index = 0
        l = 0
        n = 0
        GPT_model.train()  #开始训练
        Map.train()

        while index < len(captions):  #每个标题循环
            batch_niber = nibers[index : index+batch_size]
            batch_caption = captions[index : index+batch_size]  #16个一组
            batch_caption_feature = []
            for niber in batch_niber:
                feature = []
                for i in niber:
                    noise = torch.randn(1024).to(device)
                    this_feature = caption_features[i] + args.noise * noise
                    seed += torch.randint(1, 10, (1,)).item()
                    seed %= args.seed_point
                    torch.manual_seed(seed)
                    feature.append(this_feature.unsqueeze(0))
                
                feature = torch.cat(feature)
                batch_caption_feature.append(feature.unsqueeze(0))

            batch_caption_feature = torch.cat(batch_caption_feature).to(device)
            batch_caption_feature = Map(batch_caption_feature)  #

            ## 对batch_caption_feature(bz, hid_size)中的每一项，采集k个其他的caption样本进行Map编码，然后对于(bz, hid_size) 和 (bz, candidate_num, hid_size)做点乘并按一定阈值做相似性二分类 pred_预测 = (bz, candidate_num)
            ## 对比学习label: 对batch_caption (bz, 1)中的每一项，GPT_tokenizer编码，然后对于(bz, hid_size) 和 (bz, candidate_num, hid_size)做点乘并按一定阈值做相似性二分类 label_关系 = (bz, candidate_num)
            ## 这样可以保持Map后的embedding和语义相似度保持一致
            ## loss_对比 = cross_entropy（pred_预测，label_关系）
            
            for i in range(len(batch_caption)):#中文
                batch_caption[i] = prefix + batch_caption[i].split('.')[0] + '.'

            #用GPT来做decoder
            token = GPT_tokenizer(batch_caption, return_tensors="pt", padding = True, truncation = True, max_length = args.max_length).to(device)
                #??这里给GPT用了capiton的feature
            output = GPT_model(**token, labels = token["input_ids"], prefix = batch_caption_feature)
            loss = output.loss  #output.decoder
            optimizer.zero_grad() #用GPT作模型，训练的参数---
            loss.backward()     #反向计算一次

            ## loss = loss+loss_对比

            optimizer.step()
            l += loss.item()  #l是batch一次的损失,一个batch是16个数据
            n += 1
            
            with torch.no_grad():
                index += batch_size  #index一组epoch中的序号
                
                if index % (batch_size * args.print_frequency) == 0:
                    torch.cuda.empty_cache()
                    #这里能看出，计算n个loss的和，n个，batchsize*frequency 打印一次；l是每次的lose和，n次
                    print("[ Epoch:", epoch, "]", index, '/', len(captions), "loss =", l/(n + 0.000001))
                    l = 0
                    n = 0
    
        torch.cuda.empty_cache()

        #训练结果保存起来
        torch.save(GPT_model.state_dict(), args.checkpoint_dir + dataset_name + f"/decoder{epoch}.pth")
        torch.save(Map.state_dict(), args.checkpoint_dir + dataset_name + f"/mlp{epoch}.pth")

######################################----每一轮都输出评估结果
        #这里开始就是从图片生成标题
        with torch.no_grad():   #使用来计算
            gts_dict = dict()
            num = 0
            
            GPT_model.eval()    #算正向不反向
            Map.eval()

            #每一个图算一次
            for i in tqdm(range(len(image_features))):  #有多少图片特征,--tqdm是进度条显示
                image_feature = image_features[i].unsqueeze(0)
                image_feature = image_feature / image_feature.norm(dim = -1, keepdim = True)
                similarity = image_feature @ caption_features_norm.T  #caption_features_norm是所有训练集+的特征值

                niber = []
                for k in range(5):  #找最近的5个特征值存起来
                    _, max_id = torch.max(similarity, dim = 1)
                    niber.append(caption_features[max_id.item()].unsqueeze(0))
                    similarity[0][max_id.item()] = 0

                niber = torch.cat(niber).unsqueeze(0)  #这里面是图片和标题最相似的5个
                niber = Map(niber.float())  #float是转换成浮点值

                #--从图片生成标题-----decoder---
                   #函数定义：caption_generation(image_feature, model, tokenizer, device):
                candidate = caption_generation(niber, GPT_model, GPT_tokenizer, device)  #每个图片的特征，都让GPT去生成标题
                gts_dict[num] = [candidate.replace(",", " ,")]  #从编码转出来的文字

                num += 1
                                                #图片对应的结果
            gts = open(args.output_dir + dataset_name + f"/result_{epoch}.json", "w")
            json.dump(gts_dict, gts)
            gts.close()
            
if __name__ == "__main__":
    args = get_args()
    main(args)          #
