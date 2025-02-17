import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 不使用中文减号
plt.rcParams['font.sans-serif'] = 'FangSong'  # 设置字体为仿宋（FangSong）

# 中文标题
plt.title("点定位平均精度对比图",fontsize=14)

# 字体字典
font_dict = dict(fontsize=12,
                 color='k',
                 family='SimHei',
                 weight='light',
                 style='italic',)

map_old = np.array(pd.read_csv("E:\BJL_毕业存档\\02摄像头朝向\\03程序\\sdvb\\result\\from_14newmodel_point_map.csv")) #没有光源信息的模型的结果
map_new = np.array(pd.read_csv("E:\BJL_毕业存档\\02摄像头朝向\\03程序\sdvb\\result\\from_99model_point_map.csv")) #有光源方向模型的结果
x = np.arange(0.5,1,0.05)
plt.plot(x,map_old[:,1],label="baseline")
plt.plot(x,map_new[:,1],label="our model")
plt.xlabel("阈值",loc='center', fontdict=font_dict)
plt.ylabel("平均精度",loc='center', fontdict=font_dict)
plt.legend()
plt.show()