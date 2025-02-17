import math
import pandas as pd
import numpy as np
import datetime

def dir_get(i,path):
# 读取Excel文件#1为角度，2为日期，3为时分，4精度，5纬度，6朝向
#     path = "D:\ISS\mask_RCNN\mask_rcnn/dir_data.xlsx"
    df = pd.read_excel(path, sheet_name='Sheet1')
    df = np.array(df)
    day_list = df[:,2]
    time_list = df[:,3]
    lon_list = df[:,4]
    lat_list = df[:,5]
    dir_list = df[:,6]

    #from jupyterlab_widgets import data
    # table = data.sheets()[0] # 打开第一张表
    # nrows = table.nrows # 获取表的行数
    # ncols = table.ncols
# for i in range(len(dir_list)):
    station=0
    day_now = day_list[i].strftime('%Y-%m-%d').split("-")
    time_now = time_list[i].strftime('%H-%M-%S').split("-")
    year = [int(day_now[0])]
    month = [int(day_now[1])]
    day = [int(day_now[2])]
    hour=[int(time_now[0])]
    min = [int(time_now[1])]
    sec = [00]
    lon = [lon_list[i]]
    lat = [lat_list[i]]
    # year=[2023]
    # month=[3]
    # day=[17]
    # hour=[12]
    # min=[3]
    # sec=[00]
    # lon=[116.2]
    # lat =[40]

    for n in range(1,2):#计算表的长度
        m=n-1
    #年积日的计算
        #儒略日 Julian day(由通用时转换到儒略日)
        JD0 = int(365.25*(year[m]-1))+int(30.6001*(1+13))+1+hour[m]/24+1720981.5

        if month[m]<=2:
            JD2 = int(365.25*(year[m]-1))+int(30.6001*(month[m]+13))+day[m]+hour[m]/24+1720981.5
        else:
            JD2 = int(365.25*year[m])+int(30.6001*(month[m]+1))+day[m]+hour[m]/24+1720981.5

        #年积日 Day of year
        DOY = JD2-JD0+1

    #N0   sitar=θ
        N0 = 79.6764 + 0.2422*(year[m]-1985) - int((year[m]-1985)/4.0)
        sitar = 2*math.pi*(DOY-N0)/365.2422
        ED1 = 0.3723 + 23.2567*math.sin(sitar) + 0.1149*math.sin(2*sitar) - 0.1712*math.sin(3*sitar)- 0.758*math.cos(sitar) + 0.3656*math.cos(2*sitar) + 0.0201*math.cos(3*sitar)
        ED = ED1*math.pi/180           #ED本身有符号

        # if lon[m] >= 0:
        #     if TimeZone == -13:
        #         dLon = lon[m] - (math.floor((lon[m]*10-75)/150)+1)*15.0
        #     else:
        #         dLon = lon[m] - TimeZone[m]*15.0   #地球上某一点与其所在时区中心的经度差
        # else:
        #     if TimeZone[m] == -13:
        #         dLon =  (math.floor((lon[m]*10-75)/150)+1)*15.0- lon[m]
        #     else:
        #         dLon =  TimeZone[m]*15.0- lon[m]
        #时差
        Et = 0.0028 - 1.9857*math.sin(sitar) + 9.9059*math.sin(2*sitar) - 7.0924*math.cos(sitar)- 0.6882*math.cos(2*sitar)
        gtdt1 = hour[m] + min[m]/60.0 + sec[m]/3600.0        #地方时
        gtdt = gtdt1 + Et/60.0
        dTimeAngle1 = 15.0*(gtdt-12)
        dTimeAngle = dTimeAngle1*math.pi/180
        latitudeArc = lat[m]*math.pi/180
    # 高度角计算公式
        HeightAngleArc = math.asin(math.sin(latitudeArc)*math.sin(ED)+math.cos(latitudeArc)*math.cos(ED)*math.cos(dTimeAngle))
    # 方位角计算公式
        CosAzimuthAngle = (math.sin(HeightAngleArc)*math.sin(latitudeArc)-math.sin(ED))/math.cos(HeightAngleArc)/math.cos(latitudeArc)
        AzimuthAngleArc = math.acos(CosAzimuthAngle)
        HeightAngle = HeightAngleArc*180/math.pi
        ZenithAngle = 90-HeightAngle
        AzimuthAngle1 = AzimuthAngleArc *180/math.pi

        if dTimeAngle < 0:
            AzimuthAngle = 180 - AzimuthAngle1
        else:
            AzimuthAngle = 180 + AzimuthAngle1

        print('太阳天顶角(deg)：%f 高度角(deg)：%f 方位角(deg)：%f ' % (ZenithAngle,HeightAngle,AzimuthAngle))
        err=dir_list[i] - AzimuthAngle
        if abs(err)>180:
            err -= np.sign(err)*360
    return err,AzimuthAngle,dir_list[i]
if "__name__"=="__main__":
    dir_get(0,"D:\ISS\mask_RCNN\mask_rcnn/dir_data.xlsx")