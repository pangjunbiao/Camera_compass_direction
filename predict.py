import copy
import math
import os  #导入操作系统模块，处理文件和目录
import time
import json


import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN, det_utils
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs
import get_direction.get_dir as g_dir


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()     #创建FPN骨干网络
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model



#不太明白
#将图像坐标转换为3D坐标的函数
def image_to_3D(rate,tan_theta):
    s=math.atan2(math.sin(rate)*math.tan(tan_theta),1)   #得到弧度值
    if s*tan_theta < 0:
        s += math.pi
        if s > math.pi:
            s -= 2*math.pi
    return s          #返回转换后的3D坐标



#同步GPU并返回当前时间
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()



#在图像上绘制视觉重心点，点之间连成一条直线
def draw_image(image, key_points):
    # draw key points
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  #将图像转化成BGR格式以适应Opencv
    x = int(key_points[0][2])   #阴影的点
    y = int(key_points[0][1])
    x_1 = int(key_points[1][2])   #物体的点
    y_1 = int(key_points[1][1])
    cv2.line(image, (x, y), (x_1, y_1), (255, 255, 0), 3)  #在两个点之间绘制一条直线
    # image = cv2.resize(image,(512,512))
    # cv2.imshow("1",image)
    # cv2.save("")


#有些公式的话一般会写在训练模型里面吗

#计算两个边界框的交并比
def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])   #第一个边界框的面积
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])   #第二个边界框的面积
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)  #交集的面积
        return S_cross/(S1+S2-S_cross)   #返回交并比



def main():
    num_classes = 2    # 不包含背景
    box_thresh = 0.4
    weights_path = "./save_test_weights/99model.pth"
    # dir_path = "D:\\ISS\\mask_RCNN\mask_rcnn\\data\\coco2017\\SOBA\\test_img/"
    dir_path = "E:\\Camera_compass_direction\\02Camera_direction\\03Program\\sdvb\\dir_data/"
    label_json_path ="E:\Camera_compass_direction\\02Camera_direction\\03Program\\sdvb\\data\\coco2017\\annotations/SBU-test00.json"
    # label_json_path = "E:\\Camera_compass_direction\\02Camera_direction\\03Program\\data\\coco2017\\annotations/SBU-test00.json"
    save_path = "E:\\Camera_compass_direction\\02Camera_direction\\03Program\\sdvb\\result\\dir_result/"
    img_name = os.listdir(dir_path)   #列出目录中所有图像文件
    img_path_list = []
    for v in img_name:
        img_path_list.append(dir_path+v)    #img_path_list = []列表中保存每个图像的完整路径
    threshold = 0.5
    # img_path="C:\\Users\\BJL\\Desktop\\mid\\2.png"
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))


    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)


    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)  #确保权重存在
    weights_dict = torch.load(weights_path, map_location='cpu')   #加载保存模型权重
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)  #加载预训练模型的权重到当前模型中
    model.to(device)    #将模型移动到指定的设备（CPU 或 GPU）上。



    # read class_indict    json
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)
    f = open("E:\\Camera_compass_direction\\02Camera_direction\\03Program\\sdvb\\result\\dir_result_err.txt", "w")

    err_dir=[]  #储存方向误差
    err_sum = 0
    # load image
    num = 0
    flag_5=0
    flag_10=0
    for img_path in img_path_list:
        # img_path = r'E:\Camera_compass_direction\02Camera_direction\03Program\sdvb\dir_data\ts008.jpg'
        # img_path = "D:\ISS\mask_RCNN\mask_rcnn\dir_data/ts2.jpg"
        # img_path = "D:\ISS\mask_RCNN\mask_rcnn\data\coco2017\\train2017\web\\web-shadow0057.jpg"
        assert os.path.exists(img_path), f"{img_path} does not exits."  #确保图像存在
        original_img = Image.open(img_path).convert('RGB')   #打开并把图像转化为RGB图像
        num+=1
        name_img=img_path[-8:-4]
        # from pil image to tensor, do not normalize image
        #转化图像为张量
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)  #增加一个批次维度

        model.eval()  # 进入验证模式
        with torch.no_grad():  #在模型评估块中不计算梯度
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)            #创建一个空张量
            #1代表1张图片 3代表通道数（通常指RGB红、绿、蓝三个通道）

            model(init_img)

            t_start = time_synchronized()  #记录开始时间
            predictions = model(img.to(device))[0]  #获取模型的预测结果   #该模型是我们将训练好的参数加入进去的模型
            t_end = time_synchronized()  #记录结束时间
            print("inference+NMS time: {}".format(t_end - t_start))

            #提取预测数据
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_mask = predictions["masks"].to("cpu").numpy()
            #选取高斯核的最大响应值

            #处理预测的掩膜已找到关键点（物体位置）
            key_points = []
            label_i=0
            for temp_mask in predict_mask:
                label_class=predict_classes[label_i]
                label_i += 1
                max_point = np.max(temp_mask[0])  #找到掩膜的最大响应点
                for k in range(len(temp_mask[0])):
                    for j in range(len(temp_mask[0,k])):
                        if temp_mask[0,k,j]==max_point:
                            key_points.append([label_class,k,j])
            # 将物体的关键点移动到地面上而不是物体的中心。
            shadow_point=[]  #存阴影点
            boj_point=[]    #存目标物点
            shadow_index = []  # 找到阴影和目标的索引
            object_index = []
            #遍历所有关键点，分类物体和阴影
            # print(key_points)
            for i in range(len(key_points)):
                if key_points[i][0] == 1:
                    object_index.append(i)
                    key_points[i][1]=int(predict_boxes[i][3])#key_points[label,y,x]  更新关键点坐标
                    boj_point.append(key_points[i])
                if key_points[i][0] == 2:
                    shadow_index.append(i)
                    shadow_point.append(key_points[i])
                    # key_points[i] = [2, int(np.mean(predict_boxes[i][1::2])), int(np.mean(predict_boxes[i][0::2]))]
                    # if i==2:
                        # key_points[i]=[0,int(np.mean(predict_boxes[i][1::2])-15),int(np.mean(predict_boxes[i][0::2]))]
            if boj_point == []:
                # print(img_path)
                continue
            #绘制points点i
            if len(predict_classes)!=0 and len(predict_mask)!=0:
                pre_mask=np.zeros((len(predict_classes),len(predict_mask[0][0]),len(predict_mask[0][0][0])))
                for points_index in range(len(key_points)):
                    x = key_points[points_index][1]
                    y = key_points[points_index][2]
                    for x_range in range(x - 10,x+10):
                        for y_range in range(y - 10, y+10):
                            if x_range<img_height and y_range<img_width:
                                pre_mask[points_index][x_range][y_range] = 1
                # predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]
                predict_mask=pre_mask   #预测的图层点用于绘图
                # draw_image(original_img,predict_mask)

                if len(predict_boxes) == 0:
                    print("没有检测到任何目标!")
                    return

                #计算角度差
                # draw_image(original_img,key_points)
                # dict_obj=[] #[[][]]二维矩阵[obj索引][关联的角度]
                boj_point=np.array(boj_point)
                best_index = np.argmax(boj_point[:,1]) #最可信的目标索引     #为什么是0
                next_best_index=[]#次优目标，用来辅助进行角度确定
                half_high = img_height//2 #半高阈值   半图像高度作为阈值
                #z=len(boj_point[0])
                #处理物体方向
                for i in range(len(boj_point)):   #这个地方感觉有点错误  为什么必须循环一次
                    if i != best_index and boj_point[i][2]>half_high:
                        next_best_index.append(i)    #这个代码没看懂  没执行
                temp_list = []#存储和best_index关联的阴影点索引
                rele_list = []#存次优目标的关联索引
                for j in shadow_index:
                    a = np.sqrt((np.sum((boj_point[best_index][1] - key_points[j][1])**2+(boj_point[best_index][2] - key_points[j][2])**2))/ (
                            (predict_boxes[object_index[best_index]][0] - predict_boxes[object_index[best_index]][2])**2+(predict_boxes[object_index[best_index]][1] -predict_boxes[object_index[best_index]][3])**2))
                    a -= compute_IOU(predict_boxes[j],predict_boxes[object_index[best_index]])
                    temp_list.append([j,a])
                # 输出预测方向
                direction_list=[]  #存储方向
                if temp_list==[]:
                    # print(img_path)
                    continue
                temp_list = np.array(temp_list)
                if len(temp_list)>=1:
                    index =np.argmin(temp_list[:,1])  #找到最小误差的索引
                    i=temp_list[index][0]   #获取阴影点的索引
                    i = int(i)
                    #计算角度
                    theta=math.atan2((key_points[i][2] - key_points[object_index[best_index]][2]), (key_points[i][1] - key_points[object_index[best_index]][1]))

                    # print(theta/3.14*180)
                    direction = image_to_3D(40,theta)  #转化为3D角度
                    direction_list.append(direction)
                dirct = direction_list[0]/3.14*180
                print("图片路径：",img_path)
                print("预测偏差角：",dirct)

                #从外部数据获取实际方向
                err,az,ora = g_dir.dir_get(num-1,"E:\\Camera_compass_direction\\02Camera_direction\\03Program\\sdvb\\dir_data1.xlsx")
                yc=az+dirct
                if yc>=360:
                    yc-=360
                if yc<0:
                    yc+=360
                err_d= abs(yc-ora)
                if abs(err_d)>180:
                    err_d=abs(err_d-360)
                err_sum += err_d

                print("预测角：",yc)
                print("实际角：",ora)
                #将结果写入文件
                # str_arr=name_img+",预测角,"+str(ora)+","+"实际角,"+str(yc)+"\n"
                # f.write(str_arr)
                key_points_list=[key_points[object_index[best_index]],key_points[i]]
                # draw_image(original_img,key_points_list)
            else:
                str_arr=str(num)+"预测偏差角"+str(0)+","+"实际偏差角："+str(0)
                f.write(str_arr)
            #计算所有角度均值的方法
            # for i in object_index:
            #     n = 0
            #     if key_points[i:,3]
            #     temp_list=[i]
            #     for j in shadow_index:
            #         if abs(
            #                 np.mean(predict_boxes[i] - predict_boxes[j]) / max(
            #                     abs(np.sum(predict_boxes[i][0::2] - predict_boxes[j][0::2])),
            #                     abs(np.sum(predict_boxes[i][1::2] - predict_boxes[j][1::2])))) < 0.5:
            #             temp_list.append(j)
            #             n += 1
            #     dict_obj.append(temp_list)
            # direction_list=[]
            # pos=[]
            # best_std = 9999
            # best_pos_list = []
            # def pair(list1, pos, index, direction_list, best_std):
            #     if index >= len(list1):
            #         temp_std = np.std(direction_list)
            #         if temp_std < best_std:
            #             best_pos = pos
            #             best_std = temp_std
            #             best_pos_list.append([best_std, copy.deepcopy(best_pos)])
            #         return
            #     for i in range(1, len(list1[index])):
            #         # if list1[index][i] in pos:
            #         #     continue
            #         direction_list.append( math.atan2((key_points[i][1] - key_points[index][1]), (key_points[i][2] - key_points[index][2])))
            #         pos.append(list1[index][i])
            #         pair(list1, pos, index + 1, direction_list, best_std)
            #         pos.pop()
            #         direction_list.pop()
            # pair(dict_obj,pos,0,direction_list,best_std)
            # best_pos_list = np.array(best_pos_list)
            # # x, y = np.unravel_index(np.argmax(heatmap[i]), heatmap[i].shape)
            # x = np.argmin(best_pos_list[:,0])
            # best_pos = best_pos_list[x][1]
            # for i in range(len(dict_obj)):
            #     x = dict_obj[i][0]
            #     y = best_pos[i]
            #     direction_list.append(
            #         math.atan2((key_points[y][1] - key_points[x][1]), (key_points[y][2] - key_points[x][2])))
            # direction=np.mean(np.array(direction_list))
            # print(direction_list)

            # 绘制预测点和box
            plot_img = draw_objs(original_img,

                                 boxes=predict_boxes,
                                 classes=predict_classes,
                                 scores=predict_scores,
                                 masks=predict_mask,
                                 category_index=category_index,
                                 line_thickness=3,
                                 font='arial.ttf',
                                 font_size=20,
                                 points=key_points_list)
            #显示图片
            #plt.imshow(plot_img)
            #plt.show()
            # 保存预测的图片结果
            # test_img = save_path+"img"+name_img+".png"
            # plot_img.save(test_img)
    # f.write(str(err_dir))
    # f.close()
    print("偏差均值:",err_sum/num)

if __name__ == '__main__':
    main()




