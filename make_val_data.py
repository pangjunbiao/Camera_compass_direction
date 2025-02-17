import json
import math

import cv2
import numpy as np
from pycocotools import mask as coco_mask

def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
    grid_y, grid_x = np.mgrid[0:size_h, 0:size_w]
    D2 = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


# 获取json里面数据
def get_json_data():
    with open('D:\ISS\mask_RCNN\mask_rcnn\data\coco2017\\annotations\\instances_train2017_old.json', 'rb') as f:  # 使用只读模型，并定义名称为f
        params = json.load(f)  # 加载json文件
        params.pop("association_anno") # 删除无用信息
        # ab_path = 'D:\ISS\mask_RCNN\mask_rcnn\data\coco2017\\train2017\\'
        # 求关键点
        params["images"] = params["images"][:100]
        i=-1
        for index in params['annotations']:
            # 绘图
            # for im in  params['images']:
            #     if im["image_id"]==index["image_id"]:
            #         im_path = ab_path+im["file_name"]
            #         image = cv2.imread(im_path)
            i+=1
            if index["image_id"]>100:
                params['annotations'].pop(i)
                continue
            w = index['width']  # 原图尺寸
            h = index['height']
            w_n = 512  # 指定的图片大w_n，h_n
            h_n = 512
            a_list = index["segmentation"]
            max_a = 0
            for i in range(len(a_list)):
                if len(a_list[i]) > max_a:
                    max_a = len(a_list[i])
                    max_i = i
            a = index["segmentation"][max_i]
            sum_a_c = 0
            sum_a_l = 0
            len_a = len(a) // 2
            for i in range(len_a):
                sum_a_c += a[i * 2]
                sum_a_l += a[i * 2 + 1]
            # 求取关键点横纵坐标
            mean_c = int(sum_a_c / len_a / w * w_n)
            mean_l = int(sum_a_l / len_a / h * h_n)
            key_points=[mean_c, mean_l]
            centr_point=index["light"][0:2]
            relation_point = index["relation"]
            # 0atan(-(y0-y1)/(x0-x1))
            light_dir = math.atan2(relation_point[1]-centr_point[1],centr_point[0]-relation_point[0])
            # 求取关键点横纵坐标
            # cv2.imshow("1", image)
            # cv2.waitKey()
            index["segmentation"] = key_points
            index["bbox"][0] = index["bbox"][0]*w_n//w
            index["bbox"][2] = index["bbox"][2]*w_n//w
            index["bbox"][1] = index["bbox"][1]*h_n//h
            index["bbox"][3] = index["bbox"][3]*h_n//h
            index["keypoints"] = key_points
            index["num_keypoints"] = 1
            index['width'] = w_n # 赋予全新的宽高
            index['height'] = h_n
            index['light_dir'] = light_dir

        # 删除冗余信息
        for img in params["images"]:
            img.pop("full_mask_path")
            img.pop("object_mask_path")
            img.pop("shadow_mask_path")
            # img.pop("shadow_mask_path")
            img['width'] = w_n
            img["height"] = h_n
    return params  # 返回修改后的内容


# 写入json文件
def write_json_data(params):
    # 使用写模式，名称定义为r
    # 其中路径如果和读json方法中的名称不一致，会重新创建一个名称为该方法中写的文件名
    with open('D:\ISS\mask_RCNN\mask_rcnn\data\coco2017\\annotations\\instances_val2017.json', 'w') as r:
        # 将dict写入名称为r的文件中
        json.dump(params, r)


# 调用两个函数，更新内容
the_revised_dict = get_json_data()
write_json_data(the_revised_dict)
