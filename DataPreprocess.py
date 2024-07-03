# -*- coding: UTF-8 -*-
import fnmatch
import arcpy
import os
from arcpy.sa import *

def Extract_Local_Monthly_DNB(OriginGlobalMonthlyDNB, AllgdbPath, StartYear, EndYear, ShapeFilePath):
    # 原始的灯光影像以月为单位，全球一张图，以tif的形式存储于每个年份命名的文件夹中
    # --OriginGlobalMonthlyDNB：原始月合成灯光数据的路径
    # --AllgdbPath：局部地区的GDB数据库汇总文件夹路径
    # --ShapeFilePath：需要处理的区域的矢量边界
    # --StartYear: 需处理数据的起始年份
    # --EndYear: 需处理数据的结束年份

    for year in range(StartYear, EndYear):
        # 打开某年的全球月合成数据库，并列出所有的tif文件
        path = os.path.join(OriginGlobalMonthlyDNB, f'{year}')
        GlobalMonthDirs = [f for f in os.listdir(path) if fnmatch.fnmatch(f, '*.tif')]
        # 打开当前年份每月的GDB作为数据处理环境
        GdbPath = f"{year}.gdb"
        arcpy.env.workspace = os.path.join(AllgdbPath, GdbPath)
        arcpy.env.overwriteOutput = True
        # 输出当前的工作环境
        print(arcpy.env.workspace)

        for name in GlobalMonthDirs:
            ras = arcpy.Raster(os.path.join(path, name))
            extractbymask = arcpy.sa.ExtractByMask(ras, ShapeFilePath, 'INSIDE')
            # 输出文件命名规则：矢量范围中的地名+年份+原始灯光数据中的月份位于[14:16]位
            savedrasname = f'{ShapeFilePath[-16:-8]}_{year}{name[14:16]}'
            extractbymask.save(savedrasname)
    return

def Smooth_Monthly_DNB(inGDBpath, outGDBpath):
    # 由于云干扰、季节变动等因素，使得月合成灯光存在时序不连续性，进行5X5邻域均值平滑
    # inGDBpath：要处理的原始数据的GDB数据库路径集合
    # outGDBpath： 输出的GDB数据库路径集合

    GDBlist = fnmatch.filter(os.listdir(inGDBpath), '*.gdb')
    for gdb in GDBlist:
        gdbpath = os.path.join(inGDBpath, gdb)
        outgdbPath = os.path.join(outGDBpath, gdb)
        if not os.path.exists(outgdbPath):
            try:
                arcpy.CreateFileGDB_management(outGDBpath, gdb)
                print(f"{outgdbPath} is created successflly!")
            except arcpy.ExecuteError:
                # 打印错误信息
                print(arcpy.GetMessages(2))

        # 设置工作空间，读取每年逐月的灯光数据
        arcpy.env.workspace = gdbpath
        montlyRaslist = arcpy.ListRasters()
        for monthlyras in montlyRaslist:
            outRasname = os.path.join(outgdbPath, monthlyras)
            # 执行焦点统计，计算5X5邻域均值
            focalstaras = arcpy.sa.FocalStatistics(monthlyras, NbrRectangle(5, 5, "CELL"), "Mean")
            # 若某值超过262，则设定为周围5X5邻域的均值，也就是焦点统计的结果
            outras = arcpy.sa.RasterCalculator([monthlyras, focalstaras], ["X", "Y"], "Con(X>262, Y, X)")
            outras.save(outRasname)
    return


if __name__ == '__main__':
    rootPath = r'E:/ShanghaiFactory/'
    outPath = r'E:/ShanghaiFactory/Shanghai_Final/'
    # 自定义Extract_Local_Monthly_DNB函数的输入参数：
    OriginGlobalMonthlyDNB = r'E:/地理数据/年-月合成灯光数据/Global_Monthcomposition/'
    AllGDBPath = os.path.join(rootPath, 'SHMonthlyCompositonGDB/')
    ShapeFilePath = os.path.join(rootPath, '上海基础空间数据/ShanghaiBous.shp')
    StartYear = 2014
    EndYear = 2015
    # 执行Extract_Local_Monthly_DNB函数
    Process_1 = Extract_Local_Monthly_DNB(OriginGlobalMonthlyDNB, AllGDBPath, StartYear, EndYear, ShapeFilePath)

    # 自定义Smooth_Monthly_DNB函数的输入参数：
    ProcessGDBPath = os.path.join(rootPath, 'SHMonthlyCompositionGDB_Preprocessed/')
    # 执行Smooth_Monthly_DNB函数
    Process_2 = Smooth_Monthly_DNB(AllGDBPath, ProcessGDBPath)