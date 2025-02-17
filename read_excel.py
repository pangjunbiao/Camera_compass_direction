import pandas as pd
import numpy as np
# 读取Excel文件#1为角度，2为日期，3为时分，4精度，5纬度，6朝向
path = "E:\BJL_毕业存档\\02摄像头朝向\\03程序\\sdvb\\dir_data.xlsx"
df = pd.read_excel(path, sheet_name='Sheet1')
df = np.array(df)
print(str(df[2][2]).split("-")[0])
# 显示数据框的前5行
# print(df.head())
