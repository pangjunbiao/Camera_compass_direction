import copy
import time

import numpy as np
import torch
from sklearn.metrics import precision_score

class Evalseg:
    '''
        self.img_ids = [] 图片id
        self.results = [] 预测结果
        self.results_file_name = results_file_name 预测结果保存路径
        self.iou_type = iou_type 判别的类别点还是分割还是bbox

    '''
    def __init__(self,coco,iou_type="segm",results_file_name="seg_results.json"):
        self.coco=copy.deepcopy(coco)
        self.img_ids = []
        self.results = []
        self.results_file_name = results_file_name
        assert iou_type in ["bbox", "segm", "keypoints"]
        self.iou_type = iou_type

    def gaussian_kernel(self,size_w, size_h, center_x, center_y, sigma):
        #生成高斯核
        grid_y, grid_x = np.mgrid[0:size_h, 0:size_w]
        D2 = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)

    def get_maxpoint(self,mask):
        # 获取最大值坐标点
        loc_max = np.argmax(mask)
        i_max = loc_max//len(mask)
        j_max = loc_max%len(mask)
        return [i_max,j_max]

    def prepare_for_coco_segmentation(self, targets, outputs,thr,flag=0):
        """将预测的结果转换成COCOeval指定的格式，针对实例分割任务"""
        # 遍历每张图像的预测结果
        def get_point(mask):
            # 获得关键点
            mask = np.array(mask)
            r, c = np.where(mask == np.max(mask))
            if len(r)>1:
                r = r[0]
                c = c[0]
            point = [int(r), int(c)]
            return point
        def get_OKS(per_point,gt_point,s=512):
            oks = np.exp(-((per_point[0] - gt_point[0]) ** 2 + (per_point[1] - gt_point[1]) ** 2) / (s))
            return oks
        self.img_ids = []
        precision_list=[] #每张图的精确率
        precision_list_1=[] #阴影精确率
        precision_list_2=[]
        criterion = torch.nn.MSELoss().cuda()
        for target, output in zip(targets, outputs):
            if len(output) == 0:
                continue

            img_id = int(target["image_id"])
            if img_id in self.img_ids:
                # 防止出现重复的数据
                continue

            self.img_ids.append(img_id)
            per_image_masks = output["masks"]
            per_image_classes = output["labels"].tolist()
            per_image_scores = output["scores"].tolist()
            gt_masks = target["masks"]
            gt_boxs = target["boxes"]
            # for thr in range(0.5,0.95,0.05):
            # thr = 0.8
            ap_map = np.zeros(len(gt_masks)) # 每个目标的目测精度
            ap_map_obj=[]
            ap_map_shadow=[]
            for j in range(len(target["labels"])):
                min_loss=9999
                precision = 0
                for i in range(len(per_image_classes)):
                    if per_image_scores[i] < 0.5:
                        continue
                    # per_mask = per_image_masks[i][0]
                    per_mask = per_image_masks[i][0] # 像素点对比计算精度
                    # per_loc = self.get_maxpoint(per_mask)
                    if per_image_classes[i] == target["labels"][j]:
                        gt_mask = gt_masks[j] # 像素点对比计算精度
                        gt_box = gt_boxs[j]
                        w = gt_box[2]-gt_box[0]
                        h = gt_box[3]-gt_box[1]
                        # gt_mask = gt_masks[j]
                        # gt_loc = self.get_maxpoint(gt_mask)

                        # 计算MSE数值
                        # temp_precision = criterion(gt_mask, per_mask) * per_image_scores[i] #精度等于重合度*预测概率
                        # if temp_precision < min_loss:
                        #     min_loss = temp_precision
                        #     ap_map[j] = min_loss

                        # 计算oks数值作为评价指标 thr为阈值可设置为0.5：0.95
                        per_point = get_point(per_mask)
                        gt_point = get_point(gt_mask)
                        temp_precision = get_OKS(per_point,gt_point,w*h)
                        if temp_precision > precision:
                            precision = temp_precision
                            if precision > thr:
                                ap = 1
                            if precision <= thr:
                                ap = 0
                            ap_map[j]=ap

            precision_list.append(np.mean(ap_map))
            for i in range(len(ap_map)):
                if target["labels"][i]==1:
                    ap_map_obj.append(ap_map[i])
                else:
                    ap_map_shadow.append(ap_map[i])
            if ap_map_obj!=[]:
                precision_list_1.append(np.mean(ap_map_obj))
            if ap_map_shadow!=[]:
                precision_list_2.append(np.mean(ap_map_shadow))
            # print(time.time()-t1)
        if flag==1:
            return precision_list_1,precision_list_2
        else:
            return precision_list