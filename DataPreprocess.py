# -*- coding: UTF-8 -*-
import fnmatch
import arcpy
import os


# projectFilepath = r'E:/ShanghaiFactory/sh_Local_Coordinate.prj'
# noinspection PyPep8Naming
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

        print(ShapeFilePath)
        count = 0
        for name in GlobalMonthDirs:
            if count < 9:
                # 通过文件名过滤原始tif
                if 'masked' in name:
                    print(name)
                    ras = arcpy.Raster(os.path.join(path, name))
                    extractbymask = arcpy.sa.ExtractByMask(ras, ShapeFilePath, 'INSIDE')
                    count = count + 1
                    extractbymask.save(arcpy.env.workspace + '\BJ' + str(year) + '0' + str(count))
            else:
                if 'masked' in name:
                    print(name)
                    ras = arcpy.Raster(path + '\\' + name)
                    extractbymask = arcpy.sa.ExtractByMask(ras, ShapeFilePath, 'INSIDE')
                    count = count + 1
                    extractbymask.save(arcpy.env.workspace + '\BJ' + str(year) + str(count))
    return


if __name__ == '__main__':
    OriginGlobalMonthlyDNB = r'E:/地理数据/年-月合成灯光数据/Global_Monthcomposition/'
    AllgdbPath = r'E:/ShanghaiFactory/SHMonthlyCompositonGDB/'
    ShapeFilePath = r'E:/ShanghaiFactory/上海基础空间数据/ShanghaiBous.shp'
    StartYear = 2014
    EndYear = 2022
    Extract_Local_Monthly_DNB(OriginGlobalMonthlyDNB, AllgdbPath, StartYear, EndYear, ShapeFilePath)
