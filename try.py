
import os    # 用于文件和目录操作
import time    # 用于时间操作
import json     # 用于处理JSON数据

import cv2    # 用于图像处理
import numpy as np   # 用于数值计算
from PIL import Image    # 用于图像处理
import matplotlib.pyplot as plt  #用于绘图
import torch   # PyTorch深度学习框架
from torchvision import transforms    # 用于图像变换

from network_files import MaskRCNN   # 导入Mask R-CNN网络
from backbone import resnet50_fpn_backbone   # 导入ResNet50 FPN骨干网络
from draw_box_utils   import draw_objs   # 导入绘制边界框的工具函数


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()  # 创建ResNet50 FPN骨干网络对象
    model = MaskRCNN(backbone,          # 创建Mask R-CNN模型实例
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model   #返回模型对象

# 同步时间函数，用于确保CUDA操作的时间同步
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

# 绘制图像函数，用于在图像上绘制关键点
def draw_image(image, key_points):
    # draw key points
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) # 将PIL图像转换为OpenCV图像格式
    for point in key_points:
        x = point[2]
        y = point[1]
        if point[0]==1:
            cv2.line(image, (x, y), (x, y + 10), (0, 0, 255), 3)# 在关键点上方绘制红色线段
            cv2.line(image, (x, y), (x + 10, y), (0, 0, 255), 3)# 在关键点右侧绘制红色线段
        elif point[0]==2:
            cv2.line(image, (x, y), (x, y - 10), (0, 0, 255), 3)# 在关键点下方绘制红色线段
            cv2.line(image, (x, y), (x - 10, y), (0, 0, 255), 3)# 在关键点左侧绘制红色线段
    cv2.imshow("1",image)  # 显示图像窗口
    cv2.waitKey()   # 等待键盘输入
def main():
    num_classes = 2  # 不包含背景
    box_thresh = 0.6  # 边界框得分阈值
    #'E:\\BJL_毕业存档\\02摄像头朝向\\02数据\\data\\coco2017'  D:\ISS\mask_RCNN\mask_rcnn\data\coco2017
    weights_path = "./save_weights/99model.pth"   # 模型权重路径
    dir_path = "E:\\BJL_毕业存档\\02摄像头朝向\\02数据\\data\\coco2017\\SOBA\\SBU-test/"
    label_json_path = 'E:\\BJL_毕业存档\\02摄像头朝向\\02数据\\data\\coco2017\\annotations/SBU-test00.json'
    save_path = "E:\\BJL_毕业存档\\\02摄像头朝向\\03程序\\sdvb\\test_img/"  # 保存路径
    img_name = os.listdir(dir_path)   # 获取图像目录下的所有文件名
    img_path_list = []
    for v in img_name:
        img_path_list.append(dir_path+v)  # 将文件名和路径组合成完整的文件路径
    threshold = 0.5
    # img_path="C:\\Users\\BJL\\Desktop\\mid\\2.png"
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights  # 加载训练权重
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)   # 将模型移动到指定设备

    # read class_indict  # 读取类别索引
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # load image  #加载程序
    num = 0
    for img_path in img_path_list:
        img_path = "E:\\BJL_毕业存档\\02摄像头朝向\\02数据\\data\\coco2017\\train2017\\COCO\\000000003134.jpg"
        assert os.path.exists(img_path), f"{img_path} does not exits."
        original_img = Image.open(img_path).convert('RGB')   # 加载图像并转换为RGB格式
        num+=1
        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])  # 定义图像转换操作
        img = data_transform(original_img)  # 将PIL图像转换为Tensor
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)    # 扩展批次维度

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)   # 用全零图像初始化模型

            t_start = time_synchronized()    # 记录开始时间
            predictions = model(img.to(device))[0]   # 进行预测
            t_end = time_synchronized()   # 记录结束时间
            print("inference+NMS time: {}".format(t_end - t_start)) # 打印推理+NMS时间

            predict_boxes = predictions["boxes"].to("cpu").numpy()  # 获取预测框
            predict_classes = predictions["labels"].to("cpu").numpy()   # 获取预测类别
            predict_scores = predictions["scores"].to("cpu").numpy()   # 获取预测得分
            predict_mask = predictions["masks"].to("cpu").numpy()      # 获取预测掩码
            #选取高斯核的最大响应值
            key_points = []    # 初始化关键点列表
            label_i=0
            for temp_mask in predict_mask:
                label_class=predict_classes[label_i]
                label_i += 1
                max_point = np.max(temp_mask[0])
                for k in range(len(temp_mask[0])):
                    for j in range(len(temp_mask[0,k])):
                        if temp_mask[0,k,j]==max_point:
                            key_points.append([label_class,k,j])  # 添加关键点

            # draw_image(original_img,key_points)
                # for t_k in range(point_loc[1]-5,point_loc[1]+5):
                #     for t_j in range(point_loc[2]-5,point_loc[2]+5):
                #         temp_mask[point_loc[0]][t_k ][t_j] = max_point
            pre_mask=np.zeros((len(predict_classes),len(predict_mask[0][0]),len(predict_mask[0][0][0])))
            for points_index in range(len(key_points)):
                x = key_points[points_index][1]
                y = key_points[points_index][2]
                for x_range in range(x - 5,x + 5):
                    for y_range in range(y - 5, y + 5):
                        pre_mask[points_index][x_range][y_range] = 1.0
                # for y_range in range(y - 3, y + 3):
                #     pre_mask[points_index][x][y_range] = 1.0
            # predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]
            predict_mask=pre_mask
            # img = draw_image(original_img,predict_mask)

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
                return
            side_list=[]
            for index in range(len(key_points)):
                if key_points[0] == 1:
                    left, top, right, bottom = predict_boxes
                    side_list.append([index,max(right-left,bottom-right)])
            shadow_index = np.argwhere(predict_classes>1)
            for index,side in side_list[:0],side_list[:1]:
                for j in shadow_index:
                    if abs(key_points[index]-key_points[j]) < side:
                        predict_boxes[3]
                        



if __name__ == '__main__':
    main()

'''


import os  # 用于文件和目录操作
import time  # 用于时间操作
import json  # 用于处理JSON数据

import cv2  # 用于图像处理
import numpy as np  # 用于数值计算
from PIL import Image  # 用于图像处理
import matplotlib.pyplot as plt  # 用于绘图
import torch  # PyTorch深度学习框架
from torchvision import transforms  # 用于图像变换

from network_files import MaskRCNN  # 导入Mask R-CNN网络
from backbone import resnet50_fpn_backbone  # 导入ResNet50 FPN骨干网络
from draw_box_utils import draw_objs  # 导入绘制边界框的工具函数

# 创建模型函数
def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()  # 创建ResNet50 FPN骨干网络对象
    model = MaskRCNN(backbone,  # 创建Mask R-CNN模型实例
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model  # 返回模型对象

# 同步时间函数，用于确保CUDA操作的时间同步
def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# 绘制图像函数，用于在图像上绘制关键点
def draw_image(image, key_points):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # 将PIL图像转换为OpenCV图像格式
    for point in key_points:
        x = point[2]
        y = point[1]
        if point[0] == 1:
            cv2.line(image, (x, y), (x, y + 10), (0, 0, 255), 3)  # 在关键点上方绘制红色线段
            cv2.line(image, (x, y), (x + 10, y), (0, 0, 255), 3)  # 在关键点右侧绘制红色线段
        elif point[0] == 2:
            cv2.line(image, (x, y), (x, y - 10), (0, 0, 255), 3)  # 在关键点下方绘制红色线段
            cv2.line(image, (x, y), (x - 10, y), (0, 0, 255), 3)  # 在关键点左侧绘制红色线段
    cv2.imshow("1", image)  # 显示图像窗口
    cv2.waitKey(0)  # 等待键盘输入

# 主函数
def main():
    num_classes = 2  # 类别数，不包含背景
    box_thresh = 0.6  # 边界框得分阈值
    weights_path = "./save_weights/99model.pth"  # 模型权重路径
    dir_path = "E:\\BJL_毕业存档\\02摄像头朝向\\02数据\\data\\coco2017\\SOBA\\SBU-test/"  # 图像目录路径
    label_json_path = 'E:\\BJL_毕业存档\\02摄像头朝向\\02数据\\data\\coco2017\\annotations/SBU-test00.json'  # 标签JSON文件路径
    save_path = "E:\\BJL_毕业存档\\02摄像头朝向\\03程序\\sdvb\\test_img/"  # 保存路径
    img_name = os.listdir(dir_path)  # 获取图像目录下的所有文件名
    img_path_list = [os.path.join(dir_path, v) for v in img_name]  # 将文件名和路径组合成完整的文件路径

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 获取设备
    print("using {} device.".format(device))  # 打印设备信息
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)  # 创建模型

    # 加载训练权重
    assert os.path.exists(weights_path), "{} file does not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location=device)
    if "model" in weights_dict:
        model.load_state_dict(weights_dict["model"])
    else:
        model.load_state_dict(weights_dict)
    model.to(device)  # 将模型移动到指定设备

    # 读取类别索引
    assert os.path.exists(label_json_path), "json file {} does not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # 加载图像
    for img_path in img_path_list:
        assert os.path.exists(img_path), f"{img_path} does not exist."
        original_img = Image.open(img_path).convert('RGB')  # 加载图像并转换为RGB格式
        data_transform = transforms.Compose([transforms.ToTensor()])  # 定义图像转换操作
        img = data_transform(original_img)  # 将PIL图像转换为Tensor
        img = torch.unsqueeze(img, dim=0)  # 扩展批次维度

        model.eval()  # 进入验证模式
        with torch.no_grad():
            t_start = time_synchronized()  # 记录开始时间
            predictions = model(img.to(device))[0]  # 进行预测
            t_end = time_synchronized()  # 记录结束时间
            print("inference+NMS time: {}".format(t_end - t_start))  # 打印推理+NMS时间

            predict_boxes = predictions["boxes"].to("cpu").numpy()  # 获取预测框
            predict_classes = predictions["labels"].to("cpu").numpy()  # 获取预测类别
            predict_scores = predictions["scores"].to("cpu").numpy()  # 获取预测得分
            predict_mask = predictions["masks"].to("cpu").numpy()  # 获取预测掩码
            key_points = []  # 初始化关键点列表
            for index, temp_mask in enumerate(predict_mask):
                label_class = predict_classes[index]
                max_point = np.max(temp_mask[0])
                for k in range(len(temp_mask[0])):
                    for j in range(len(temp_mask[0, k])):
                        if temp_mask[0, k, j] == max_point:
                            key_points.append([label_class, k, j])  # 添加关键点

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
                continue  # 如果没有检测到目标，跳过当前循环的剩余部分

            side_list = []
            for index, key_point in enumerate(key_points):
                if key_point[0] == 1:
                    left, top, right, bottom = predict_boxes[index]
                    side = max(right - left, bottom - top)
                    side_list.append([index, side])  # 添加边长

            shadow_index = np.argwhere(predict_classes > 1).flatten()  # 获取非背景类的索引
            for index, side in side_list:
                for j in shadow_index:
                    if abs(key_points[index][1] - key_points[j][1]) < side and \
                       abs(key_points[index][2] - key_points[j][2]) < side:
                        # 如果检测到的点与shadow_index中的点足够接近，则执行某些操作
                        pass  # 使用 pass 作为占位符，实际代码中需要替换为具体操作

if __name__ == '__main__':
    main()  # 运行主函数
'''

