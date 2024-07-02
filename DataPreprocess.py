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

def Judge_ExtremeNTL(inGDBpath, Buffersize):
    # 判断每个工厂的缓冲区内是否存在极端大的灯光值
    # inGDBpath: 要处理的原始数据的GDB数据库路径集合
    #
    for year in range(14, 22):
        gdbpath = "\SH_VIIRS_20" + str(year) + "年月合成数据.gdb"
        arcpy.env.workspace = inGDBpath + gdbpath
        arcpy.env.overwriteOutput = True
        print('Processing................. ')
        Industrial = ''
        POI = ''
        for shp in arcpy.ListFeatureClasses('', ''):
            if 'SH_euluc2018' in shp:
                Industrial = shp
                print(Industrial)
            if '工厂POI' in shp:
                POI = shp
                print(POI)

        for ras in arcpy.ListRasters('', ''):
            if 'SH20' in ras:
                # 构建工厂POI附近缓冲区
                POI_Buffer = arcpy.CreateScratchName("POIbuffer", data_type="FeatureClass", workspace="in_memory")
                arcpy.Buffer_analysis(POI, POI_Buffer, Buffersize)
                # 每年的灯光数据转栅格
                Ras = arcpy.Raster(ras)
                temp_ras = arcpy.CreateScratchName("ras_prj", data_type="FeatureClass", workspace="in_memory")
                arcpy.Raster(arcpy.ProjectRaster_management(Ras, temp_ras, projectFilepath))
                # 计算缓冲区内的最大灯光值
                POIbuffer_temp_table = arcpy.CreateScratchName("POIbuffertable", data_type="Table",
                                                               workspace="in_memory")
                print(POIbuffer_temp_table)
                arcpy.sa.ZonalStatisticsAsTable(POI_Buffer, 'ID', temp_ras, POIbuffer_temp_table,
                                                statistics_type='最大值')
                # outExcel = r"C:\Users\KJ\Documents\ShanghaiFactory\上海月度统计"
                # arcpy.TableToExcel_conversion(temp_table, outExcel + '\\'+ str(rasBJ) + ".xlsx",
                # Use_field_alias_as_column_header="ALIAS",
                # Use_domain_and_subtype_description="CODE")
                POI_temp_table = arcpy.CreateScratchName("SH_POItable", data_type="Table", workspace="in_memory")
                arcpy.sa.ZonalStatisticsAsTable(POI, 'ID', temp_ras, POI_temp_table, statistics_type='最大值')
                # print(POI_temp_table)

                # 为统计的结果新增字段
                arcpy.AddField_management(POI_temp_table, 'MAXNTLCheck', 'SHORT',
                                          field_is_nullable="NULLABLE")  # 判断缓冲区内是否存在超过工厂POI灯光两倍的灯光
                arcpy.AddField_management(POI_temp_table, 'BufferMAX', 'DOUBLE',
                                          field_is_nullable="NULLABLE")  # 存储缓冲区内的最大灯光值
                arcpy.AddField_management(POIbuffer_temp_table, 'MAXNTLCheck', 'SHORT',
                                          field_is_nullable="NULLABLE")

                # 对比工厂POI灯光亮度与缓冲区内最大灯光亮度，若超过其2倍及以上，则认为该工厂灯光受周围较大灯光的影响，‘MAXNTLCheck’赋值为1，否则为0
                POIshpField = ['ID', 'MAX', 'MAXNTLCheck', 'BufferMAX']
                POIbuffershpField = ['ID', 'MAX', 'MAXNTLCheck']
                POIRows = arcpy.UpdateCursor(POI_temp_table, POIshpField)
                POIbufferRows = arcpy.UpdateCursor(POIbuffer_temp_table, POIbuffershpField)
                while True:
                    poirow = POIRows.next()
                    if not poirow:
                        break

                    while True:
                        poibufferrow = POIbufferRows.next()
                        if not poibufferrow:
                            break

                        if poibufferrow.ID == poirow.ID:
                            if poibufferrow.MAX > poirow.MAX * 2:
                                # print('The poi ID: %d, poimax = %.3f;The poibuffer ID: %d, poibuffermax = %.3f'
                                # %(poirow.ID,poirow.MAX,poibufferrow.ID,poibufferrow.MAX))
                                poirow.setValue('MAXNTLCheck', 1)
                                poibufferrow.setValue('MAXNTLCheck', 1)
                                # POIRows.updateRow(poirow)

                                # 将工厂缓冲区内的灯光最大值写入工厂统计table中
                                poirow.setValue('BufferMAX', poibufferrow.MAX)
                                POIRows.updateRow(poirow)
                                POIbufferRows.updateRow(poibufferrow)
                            else:
                                # print('The poi ID: %d is real NTL, and MAX = %.3f' %(poirow.ID,poirow.MAX))
                                poirow.setValue('MAXNTLCheck', 0)
                                poibufferrow.setValue('MAXNTLCheck', 0)
                                # POIRows.updateRow(poirow)
                                poirow.setValue('BufferMAX', poibufferrow.MAX)
                                POIRows.updateRow(poirow)
                                POIbufferRows.updateRow(poibufferrow)
                            break
                        else:
                            print('The ID %d is not matched!' % poibufferrow.ID)
                            # poirow.setValue('MAXNTLCheck', 99)
                            # POIRows.updateRow(poirow)

                # 新生成的POI Buffer与zonal统计的结果连接，将连接后的buffer图层存入gdb，为后续identity和calculate geometry做准备
                SH_POI_Buffer1000_out = arcpy.CreateFeatureclass_management(TempshpPath, str(Ras) + "MAXbuffer",
                                                                            "POLYGON")  # , spatial_reference=projectFilepath
                arcpy.Buffer_analysis(POI, SH_POI_Buffer1000_out, Buffersize)
                arcpy.management.JoinField(SH_POI_Buffer1000_out, 'ID', POI_temp_table, 'ID')
                outExcel = outExcelPath + str(Ras) + '_buffer_MAXNTLCheck.xlsx'
                print(outExcel)
                arcpy.conversion.TableToExcel(SH_POI_Buffer1000_out, outExcel)
                print('The %s max result is processed!' % Ras)

                # 关闭所有的临时内存文件
                arcpy.Delete_management(POI_Buffer)
                arcpy.Delete_management(temp_ras)
                arcpy.Delete_management(POIbuffer_temp_table)
                arcpy.Delete_management(POI_temp_table)


if __name__ == '__main__':
    rootPath = r'E:/ShanghaiFactory/'
    # 自定义Extract_Local_Monthly_DNB函数的输入参数：
    OriginGlobalMonthlyDNB = r'E:/地理数据/年-月合成灯光数据/Global_Monthcomposition/'
    AllGDBPath = os.path.join(rootPath, 'SHMonthlyCompositonGDB/')
    ShapeFilePath = os.path.join(rootPath, '上海基础空间数据/ShanghaiBous.shp')
    StartYear = 2022
    EndYear = 2023
    # 执行Extract_Local_Monthly_DNB函数
    Process_1 = Extract_Local_Monthly_DNB(OriginGlobalMonthlyDNB, AllGDBPath, StartYear, EndYear, ShapeFilePath)

    # 自定义Smooth_Monthly_DNB函数的输入参数：
    ProcessGDBPath = os.path.join(rootPath, 'SHMonthlyCompositionGDB_Preprocessed/')
    # 执行Smooth_Monthly_DNB函数
    Process_2 = Smooth_Monthly_DNB(AllGDBPath, ProcessGDBPath)

    # 自定义Judge_ExtremeNTL函数的输入参数：
    Judge_ExtremeNTL(AllGDBPath, )



