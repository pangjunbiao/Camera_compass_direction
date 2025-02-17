import math
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch
import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric
from seg_Eval import Evalseg
import matplotlib.pyplot as plt


def draw_image(image, key_points):
    # draw key points
    image=np.array(image)
    image = np.transpose(image, (1,2,0))
    image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    for i in range(len(key_points)):
        x = key_points[i][0]
        y = key_points[i][1]

        cv2.line(image, (x, y), (x, y + 5), (0, 0, 255), 3)
        cv2.line(image, (x, y), (x + 5, y), (0, 0, 255), 3)
                # elif i == 1:
                #     cv2.line(image, (x, y), (x, y + 10), (33, 33, 133), 3)
                #     cv2.line(image, (x, y), (x - 10, y), (33, 33, 133), 3)
    cv2.imshow("1",image)
    cv2.waitKey(0)


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # o = time.ctime()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # key_points = []
        # for t_mask in targets[0]['masks']:
        #     flag=0
        #     for k in range(len(t_mask)):
        #         if flag==1:
        #             break
        #         for j in range(len(t_mask[k])):
        #             if t_mask[k][j] == 1:
        #                 key_points.append([k, j])
        #                 flag=1
        #                 break
        # draw_image(images[0],key_points)
        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)                                      #标记一下  这边loss

            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        # p = time.ctime()
        # print('eptime',p)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device,type="train"):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    det_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="bbox", results_file_name="det_results.json")
    key_metric = Evalseg(data_loader.dataset.coco, iou_type="segm", results_file_name="seg_results.json")
    # key_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="segm", results_file_name="seg_results.json")
    seg_info = []
    key_point_info_1=[]
    key_point_info_2=[]
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        det_metric.update(targets, outputs)
        seg_info.extend(key_metric.prepare_for_coco_segmentation(targets, outputs,thr=0.95)) #输入不同的阈值表示点预测偏差的准确程度
        key_1,key_2 = key_metric.prepare_for_coco_segmentation(targets, outputs,thr=0.95,flag=1)
        key_point_info_1.extend(key_1)
        key_point_info_2.extend(key_2)
        # key_metric.update(targets, outputs)
        metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 同步所有进程中的数据
    det_metric.synchronize_results()
    # key_metric.synchronize_results()

    if utils.is_main_process():
        coco_info = det_metric.evaluate()

        # seg_info = 1
    else:
        coco_info = None
        seg_info = None
    if type=="train":
        return coco_info,seg_info
    else:
        return coco_info,seg_info,key_point_info_1,key_point_info_2

