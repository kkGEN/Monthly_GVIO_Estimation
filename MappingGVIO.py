import os
import time
import arcpy
import pandas as pd
import numpy as np
import functools


def Time_Decorator(func):
    # 输出函数运行时间的修饰器
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f'Start_time: {start_time_str}.')
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'Excution_time: {end_time-start_time}.')
        return result
    return wrapper
def Check_GDBPath_Exist(GDBFullPath, GDBFolder, GDBName):
    # 检查GDB数据库是否存在，若不存在则新建
    if not os.path.exists(GDBFullPath):
        try:
            arcpy.CreateFileGDB_management(GDBFolder, GDBName)
            print(f"{GDBFullPath} is created successfully!")
        except arcpy.ExecuteError:
            # 打印错误信息
            print(arcpy.GetMessages(2))
    return

def Check_FolderPath_Exist(outFolderPath):
    if not os.path.exists(outFolderPath):
        os.makedirs(outFolderPath)
        print(f'{outFolderPath} is created successflly!')
    return

@Time_Decorator
def Mask_Extract(inmask, ingdb, resultgdb):
    # Step 1: 利用水体和缓冲区掩膜，将原始的月合成数据进行裁剪
    # inmask：水体和工业用地矢量掩膜
    # ingdb: 预处理后的原始灯光数据库
    # resultgdb：掩膜后的结果数据库


    for year in range(2014, 2023):
        arcpy.env.workspace = os.path.join(ingdb, f'{year}.gdb')
        arcpy.env.overwriteOutput = True
        # 读取每一个栅格图层，进行掩膜
        for ras in arcpy.ListRasters():
            Ras = arcpy.Raster(ras)
            outMasked = arcpy.sa.ExtractByMask(Ras, inmask, "INSIDE")
            outMasked.save(os.path.join(resultgdb, ras))
        print('Mask of Shanghai_{} is done!'.format(year))

    return

@Time_Decorator
def Relocate_GVIO(estimated_result, maskedgdb, GVIOgdb):
    # Step 2: 将预测的工业产值分配到灯光影像上
    # estimated_result: SA-CNN计算得到的最佳GVIO结果
    # maskedgdb：经过掩膜的灯光影像
    # GVIOgdb：存储GVIO空间化后的结果路径

    # 读取最佳的GVIO预测结果
    df_estimate = pd.read_excel(estimated_result)
    arcpy.env.workspace = maskedgdb
    for ras in arcpy.ListRasters():
        # 获取每个月份的名称
        estimate_row = int(ras[-6:])
        # 获取每月的估算结果
        month_row = df_estimate.loc[df_estimate['Month'] == estimate_row]
        if month_row is not None:
            # 获取每月的预测值
            total_value = month_row.Pred.iloc[0]
            raster = arcpy.Raster(ras)
            # 将灯光数据转换为数组，并求和
            rasterArray = arcpy.RasterToNumPyArray(raster, nodata_to_value=0)  # np.nan
            raster_sum = np.nansum(rasterArray)
            # 根据POI位置获取的工厂灯光强度（均值，或众数，或95%分位数），大于阈值的灯光像素不再参与分配工业产值
            # 这里的阈值目前是取的论文中的95%分位数，即42.2
            raster_perc_95 = 42.2
            # 更改Con表达式，实现不同灯光值的截断
            con_raster = arcpy.sa.Con(raster > raster_perc_95, 0, raster)

            # 设置输出GDB为工作空间
            arcpy.env.workspace = GVIOgdb
            arcpy.env.overwriteOutput = True

            # 重新分配工业产值到每个像素
            output_raster = con_raster * (total_value / raster_sum)
            output_raster_path = os.path.join(GVIOgdb, ras)
            output_raster.save(output_raster_path)
            print('Relocated GVIO of {} is done!'.format(ras))
            # 将工作空间切换回输入灯光数据库
            arcpy.env.workspace = maskedgdb

    return


if __name__ == "__main__":
    root = r'E:/ShanghaiFactory/'
    basicgeo = os.path.join(root, 'SHBasicGeoData')
    results = os.path.join(root, 'Shanghai_Final')
    InGDBPath = os.path.join(root, 'SHMonthlyCompositionGDB_Preprocessed')
    BufferSize = '1500 METERS'  # <<Caution!!!>> 缓冲区的距离，这是一个可变参数，可选500m,1000m,1500m,2000m

    # 设置Mask_Extract函数的输入参数
    GVIOResultRoot = os.path.join(root, 'SpatialResultofGVIO')
    Check_FolderPath_Exist(GVIOResultRoot)
    MaskoutGDBName = f'Masked_NTL_{BufferSize}.gdb'  # 存放掩膜后的灯光数据结果
    MaskoutGDBPath = os.path.join(GVIOResultRoot, MaskoutGDBName)
    Check_GDBPath_Exist(MaskoutGDBPath, GVIOResultRoot, MaskoutGDBName)  # 检查路径是否存在，不存在则新建
    MaskLayer = os.path.join(basicgeo, 'Shanghai_GVIO_Mask.shp')
    # 执行Mask_Extract函数
    Mask_Extract(MaskLayer, InGDBPath, MaskoutGDBPath)

    # 设置Relocate_GVIO函数的输入参数
    outCNNResult = os.path.join(results, f'Step05_SA-CNN_Results/{BufferSize}_Result.xlsx')
    GVIO_GDBName = f'Spatial_GVIO_{BufferSize}.gdb'  # 存放空间化后的GVIO结果
    GVIO_GDBPath = os.path.join(GVIOResultRoot, GVIO_GDBName)
    Check_GDBPath_Exist(GVIO_GDBPath, GVIOResultRoot, GVIO_GDBName)
    # 执行Relocate_GVIO函数
    Relocate_GVIO(outCNNResult, MaskoutGDBPath, GVIO_GDBPath)
