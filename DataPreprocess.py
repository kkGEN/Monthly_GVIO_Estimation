import arcpy
import os

OriginGlobalMonthlyDNB = r'E:/ShanghaiFactory/Monthcomposition/'
AllgdbPath = r"E:/ShanghaiFactory/SHMonthlyCompositonGDB/"
# projectFilepath = r'E:/ShanghaiFactory/sh_Local_Coordinate.prj'

for year in range(2021, 2022):
    #open某年的全球月合成
    path = OriginGlobalMonthlyDNB + str(year)
    globalmonth = os.listdir(path)
    #open某年每月的GDB作为数据处理环境
    gdbpath = 'SH_VIIRS_' + str(year) + '年月合成数据.gdb'
    arcpy.env.workspace = AllgdbPath + gdbpath
    arcpy.env.overwriteOutput = True
    print(arcpy.env.workspace)
    BJShp = ''
    for shp in arcpy.ListFeatureClasses('', ''):
        if 'Shanghaibous' in shp:
            BJShp = shp
            print(BJShp)
    print('............Processing............. ')
    count = 0
    for name in globalmonth:
        if count < 9:
            #通过文件名过滤原始tif
            if 'masked' in name:
                print(name)
                ras = arcpy.Raster(path + '\\' +name)
                extractbymask = arcpy.sa.ExtractByMask(ras, BJShp, 'INSIDE')
                count = count + 1
                extractbymask.save(arcpy.env.workspace+'\BJ'+str(year)+'0'+str(count))
        else:
            if 'masked' in name:
                print(name)
                ras = arcpy.Raster(path + '\\' +name)
                extractbymask = arcpy.sa.ExtractByMask(ras, BJShp, 'INSIDE')
                count = count + 1
                extractbymask.save(arcpy.env.workspace+'\BJ'+str(year)+str(count))
