import os
import json

####---把结果放到mscoco_test.json里面生成maigc用的评测结果

#其他项目中的测试数据
in_path = r'./mscoco_test.json'
with open(in_path, 'r') as f:           #读数据
    test_content = json.load(f)

#其他项目中的测试数据
in_path = r'./result_0.json'            #读一个结果
with open(in_path, 'r') as f:           #读数据
    result_one_content = json.load(f)   #结果中的每一个

result_num=0

import json
save_path = r'./mscoco_result_knight.json'
with open(save_path, 'r') as f:           #读数据
    save_content = json.load(f)

#找到元素中的某一个，
for item in test_content:
    #print('循环的当前图片是：'+item["image_name"])
    result_str = str(result_num)
    item["prediction"] = result_one_content[result_str][0]
    result_num += 1

with open(save_path, 'w') as outfile:  # 向数据库写入数据
    json.dump(test_content, outfile, indent=4)  # 将python对象保存成json格式，indent表示缩进的空格数





