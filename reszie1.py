#coding=utf-8
import os
import cv2
DATADIR = "D:\ISS\mask_RCNN\mask_rcnn\dir_data/"
IMG_SIZE=512
path=os.path.join(DATADIR)
img_list=os.listdir(path)
ind=0

for i in img_list:
    img_array=cv2.imread(os.path.join(path,i),cv2.IMREAD_COLOR)
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    img_name = i
    save_path: str = DATADIR+i
    ind=ind+1
    cv2.imwrite(save_path,new_array)
    # break

# DATADIR = "D:\\ISS\\yolox\\yolox\\data\\test_img_2"
# IMG_SIZE=512
# path=os.path.join(DATADIR)
# img_list=os.listdir(path)
# ind=0
#
# # if not os.path.exist("path_to_root"):
# #     os.makedirs("path_to_root"+DATADIR)
# for i in img_list:
#     img_array=cv2.imread(os.path.join(path,i),cv2.IMREAD_COLOR)
#     new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
#     img_name = i
#     save_path = DATADIR+i
#     ind=ind+1
#     cv2.imwrite(save_path,new_array)