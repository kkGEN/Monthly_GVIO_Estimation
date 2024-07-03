# -*- coding: UTF-8 -*-
import fnmatch
import arcpy
import os
from arcpy.sa import *

def Judge_Extreme_NTL(inGDBpath, projectFilepath, outExcelPath, Buffersize, StartYear, EndYear, POI, OutshpPath):
    # 判断每个工厂的缓冲区内是否存在极端大的灯光值
    # inGDBpath: 要处理的原始数据的GDB数据库路径集合
    # outExcelPath: 每一个缓冲区距离下的输出excel存储文件夹路径
    # Buffersize：自定义缓冲区距离
    # POI: 输入中类为”工厂“的POI
    # OutshpPath: 输出工厂POI的缓冲区，并存储了是否存在极端灯光值信息
    # StartYear：需处理数据的起始年份
    # EndYear：需处理数据的结束年份

    for year in range(StartYear, EndYear):
        # 按照年份读取（经过平滑的）地区月度灯光数据，并以输入数据所在数据库为工作空间
        GdbPath = f"{year}.gdb"
        arcpy.env.workspace = os.path.join(inGDBpath, GdbPath)
        arcpy.env.overwriteOutput = True
        # 逐月读取栅格数据进行处理
        for ras in arcpy.ListRasters():
            # 灯光数据读取并转为栅格
            Ras = arcpy.Raster(ras)
            # （临时文件）根据POI的定位，构建逐个工厂的缓冲区
            POI_Buffer = arcpy.CreateScratchName("POIbuffer", data_type="FeatureClass", workspace="in_memory")
            arcpy.Buffer_analysis(POI, POI_Buffer, Buffersize)
            # （临时文件）计算每个工厂对应位置的灯光值，这里的statistics_type取最大值，均值，中值等均为同一个值
            POI_temp_table = arcpy.CreateScratchName("SH_POItable", data_type="Table", workspace="in_memory")
            arcpy.sa.ZonalStatisticsAsTable(POI, 'ID', Ras, POI_temp_table, statistics_type='最大值')
            # （临时文件）计算每个缓冲区内的最大灯光值
            POIbuffer_temp_table = arcpy.CreateScratchName("POIbuffertable", data_type="Table", workspace="in_memory")
            arcpy.sa.ZonalStatisticsAsTable(POI_Buffer, 'ID', Ras, POIbuffer_temp_table, statistics_type='最大值')

            # 为统计的结果新增字段：
            # 工厂灯光值tale-’MAXNTLCheck‘：判断缓冲区内是否存在超过工厂POI灯光两倍的灯光（0或1）
            arcpy.AddField_management(POI_temp_table, 'MAXNTLCheck', 'SHORT', field_is_nullable="NULLABLE")
            # 工厂灯光值tale-'BufferMAX'：存储每个工厂缓冲区内的最大灯光值
            arcpy.AddField_management(POI_temp_table, 'BufferMAX', 'DOUBLE', field_is_nullable="NULLABLE")
            # 工厂缓冲区内灯光值tale-’MAXNTLCheck‘：判断缓冲区内是否存在超过工厂POI灯光两倍的灯光（0或1）
            arcpy.AddField_management(POIbuffer_temp_table, 'MAXNTLCheck', 'SHORT', field_is_nullable="NULLABLE")

            # 存储字段名称，方便后续对应读取
            POIshpField = ['ID', 'MAX', 'MAXNTLCheck', 'BufferMAX']
            POIbuffershpField = ['ID', 'MAX', 'MAXNTLCheck']
            # 获取更新游标
            POIRows = arcpy.UpdateCursor(POI_temp_table, POIshpField)
            POIbufferRows = arcpy.UpdateCursor(POIbuffer_temp_table, POIbuffershpField)
            # 对两个表格进行匹配和比较，判断缓冲区内是否存在极端值，并check
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
                            poirow.setValue('MAXNTLCheck', 1)
                            poibufferrow.setValue('MAXNTLCheck', 1)

                            # 将工厂缓冲区内的灯光最大值写入工厂统计table中
                            poirow.setValue('BufferMAX', poibufferrow.MAX)
                            POIRows.updateRow(poirow)
                            POIbufferRows.updateRow(poibufferrow)
                        else:
                            poirow.setValue('MAXNTLCheck', 0)
                            poibufferrow.setValue('MAXNTLCheck', 0)

                            # 将工厂缓冲区内的灯光最大值写入工厂统计table中
                            poirow.setValue('BufferMAX', poibufferrow.MAX)
                            POIRows.updateRow(poirow)
                            POIbufferRows.updateRow(poibufferrow)
                        break
                    else:
                        print('The ID %d is not matched!' % poibufferrow.ID)

            # 新生成的POI Buffer与zonal统计的结果连接，将连接后的buffer图层存入gdb，为后续identity和calculate geometry做准备
            SH_POI_Buffer_out = arcpy.CreateFeatureclass_management(OutshpPath, f"{Ras}MAXbuffer", "POLYGON")
            arcpy.Buffer_analysis(POI, SH_POI_Buffer_out, Buffersize)
            arcpy.JoinField_management(SH_POI_Buffer_out, 'ID', POI_temp_table, 'ID')
            # 将统计的临时table存储为excel（作为备份后续步骤不会使用）
            outExcel = os.path.join(outExcelPath, f'{Ras}_buffer_MAXNTLCheck.xlsx')
            arcpy.TableToExcel_conversion(SH_POI_Buffer_out, outExcel)
            print('The {} buffer max result is processed!'.format(Ras))

            # 关闭所有的临时内存文件
            arcpy.Delete_management(POI_Buffer)
            arcpy.Delete_management(POIbuffer_temp_table)
            arcpy.Delete_management(POI_temp_table)
    return

def Check_GDBPath_Exist(GDBFullPath, GDBFolder, GDBName):
    # 检查GDB数据库是否存在，若不存在则新建
    if not os.path.exists(GDBFullPath):
        try:
            arcpy.CreateFileGDB_management(GDBFolder, GDBName)
            print(f"{GDBFullPath} is created successflly!")
        except arcpy.ExecuteError:
            # 打印错误信息
            print(arcpy.GetMessages(2))
    return
def Check_FolderPath_Exist(outFolderPath):
    if not os.path.exists(outFolderPath):
        os.makedirs(outFolderPath)
        print(f'{outFolderPath} is created successflly!')
    return

def Sumup_Each_Landuse_NTL(RasGDBpath, InGDBPath, BuffInterPath, BuffInterF2PPath, StartYear, EndYear, Industrial):
    # RasGDBpath：灯光数据数据库GDB
    # InGDBPath: 存储工厂缓冲区矢量范围的GDB
    # StartYear:
    # EndYear:
    # Industrial: Euluc土地利用矢量


    # 循环打开每年的月合成灯光数据库
    for year in range(StartYear, EndYear):
        GdbPath = f"{year}.gdb"
        arcpy.env.workspace = os.path.join(RasGDBpath, GdbPath)
        # 打开每月的灯光栅格影像
        for ras in arcpy.ListRasters():
            rasSH = arcpy.Raster(ras)
            # 列出带缓冲区的工厂POI图层
            arcpy.env.workspace = InGDBPath
            arcpy.env.overwriteOutput = True
            BufferShpList = arcpy.ListFeatureClasses()
            for BufferShp in BufferShpList:
                if f'{rasSH}' in BufferShp:
                    # 输出buffer与EULUC相交的结果
                    BufferInterOutPath = os.path.join(BufferIntersectPath, f'{BufferShp}_Intersect')
                    arcpy.Intersect_analysis([[BufferShp, 1], [Industrial, 2]], BufferInterOutPath, "ALL")
                    print(f'{BufferShp}_inter is done!')

                    # 为相交处理后的矢量图层增加主键字段，方便后续的字段计算
                    # 判断"BufferInterID"是否存在，不存在则新建
                    field_to_check = "BufferInterID"
                    fields = arcpy.ListFields(BufferInterOutPath)
                    # next() 函数尝试从生成器表达式中获取第一个 True 值，如果没有找到，则返回默认值 False
                    field_exists = next((True for field in fields if field.name == field_to_check), False)
                    if not field_exists:
                        arcpy.AddField_management(BufferInterOutPath, field_to_check, 'LONG', 9)
                        print(f"Field '{field_to_check}' not exist, added now!")
                    else:
                        print(f"Field '{field_to_check}' already exists!")
                    # 计算
                    expression = 'Increment()'
                    codeblock = '''
                        rec = 0
                        def Increment():
                            global rec
                            Start=1
                            Interval=1
                            if(rec == 0):
                                rec=Start
                            else:
                                rec=rec+Interval
                            return rec
                        '''
                    arcpy.CalculateField_management(BufferInterOutPath, 'BufferInterID', expression, "PYTHON3", codeblock)

                    # 筛选面积大于3E-06的斑块，并输出为excel
                    arcpy.MakeFeatureLayer_management(BufferInterOutPath, "lyr")
                    arcpy.SelectLayerByAttribute_management("lyr", "NEW_SELECTION", "Shape_Area >= 3E-06")
                    arcpy.TableToExcel_conversion("lyr", f'{BuffInterPath}{BufferShp}.xlsx')
                    print(f'{BufferShp} Intersect Done!')

                    # 将添加主键后的矢量图层转换为点Points
                    BuffInterF2POutPath = f'{BuffInterF2PPath}{BufferShp}Inter_F2P'
                    # 将按面积筛选的BufferIntersect图层转换为点
                    arcpy.FeatureToPoint_management("lyr", BuffInterF2POutPath)
                    print(f'{BufferShp} Feature_to_Point Done!')

                    # 根据土地利用类型算后的斑块面积总和
                    LUNTL_Temp_table = arcpy.CreateScratchName("SH_POItable", data_type="Table", workspace="in_memory")
                    arcpy.sa.ZonalStatisticsAsTable(BuffInterF2POutPath, 'BufferInterID', rasSH, LUNTL_Temp_table)
                    arcpy.TableToExcel_conversion(LUNTL_Temp_table, BufferInterZonalPath + str(BufferShp) + '.xlsx')
                    print(f'{BufferShp} Zonal_as_table Done!')
            # 切换回栅格图层所在数据库
            arcpy.env.workspace = os.path.join(RasGDBpath, GdbPath)


if __name__ == '__main__':
    rootPath = r'E:/ShanghaiFactory/'
    outPath = r'E:/ShanghaiFactory/Shanghai_Final/'
    StartYear = 2014
    EndYear = 2023

    # 自定义Judge_ExtremeNTL函数的输入参数：
    Judge_ExtreNTL_Folder = os.path.join(outPath, 'Step01_Judge_Extreme_NTL/')
    ProjectFilePath = os.path.join(rootPath, 'sh_Local_Coordinate.prj')
    POIsIndustrialPath = os.path.join(rootPath, '上海基础空间数据/Shanghai_POI_Industrial.shp')
    ProcessGDBPath = os.path.join(rootPath, 'SHMonthlyCompositionGDB_Preprocessed/')

    BufferSize = '1500 METERS'  # <<Caution!!!>> 缓冲区的距离，这是一个可变参数，可选500m,1000m,1500m,2000m
    # 存放Buffer之后的excel版本
    outExcelFolderPath = os.path.join(Judge_ExtreNTL_Folder, f'Judge_Extreme_NTL_{BufferSize}')
    Check_FolderPath_Exist(outExcelFolderPath)
    BufferGDBName = f'Judge_Extreme_NTL_Buffer_{BufferSize}.gdb'  # 存放该步骤得到的缓冲区结果，结果保留了缓冲区内是否存在极端值信息
    BufferGDBPath = os.path.join(Judge_ExtreNTL_Folder, BufferGDBName)
    Check_GDBPath_Exist(BufferGDBPath, Judge_ExtreNTL_Folder, BufferGDBName)  # 检查路径是否存在，不存在则新建

    # 执行Judge_Extreme_NTL函数
    Process_3 = Judge_Extreme_NTL(ProcessGDBPath, ProjectFilePath, outExcelFolderPath, BufferSize,
                                  StartYear, EndYear, POIsIndustrialPath, BufferGDBPath)

    # 自定义Sumup_Each_Landuse_NTL函数的输入参数：
    Sumup_Each_Landuse_NTL_Folder = os.path.join(outPath, 'Step02_Sumup_Each_Landuse_NTL/')
    # 存放Buf经过Intersect的excel版本
    OutInteExcelPath = os.path.join(Sumup_Each_Landuse_NTL_Folder, f'Sumup_Each_Landuse_NTL_BufInte_{BufferSize}')
    Check_FolderPath_Exist(OutInteExcelPath)
    # 存放Buf经过Intersect后FeatureToPoints的excel版本
    OutInteF2PExcelPath = os.path.join(Sumup_Each_Landuse_NTL_Folder, f'Sumup_Each_Landuse_NTL_BufInteF2P_{BufferSize}')
    Check_FolderPath_Exist(OutInteF2PExcelPath)
    # 土地利用数据路径
    EulucPath = os.path.join(rootPath, '上海基础空间数据/Shanghai_Euluc_2018.shp')
    # 存放Intersect后的GDB路径
    BufferIntersectGDBName = f'Sumup_Each_Landuse_NTL_BufInte_{BufferSize}.gdb'
    BufferIntersectPath = os.path.join(Sumup_Each_Landuse_NTL_Folder, BufferIntersectGDBName)
    Check_GDBPath_Exist(BufferIntersectPath, Sumup_Each_Landuse_NTL_Folder, BufferIntersectGDBName)
    # 存放Intersect后并FeatureToPoints的GDB路径
    BufferInterFeature2PointName = f'Sumup_Each_Landuse_NTL_BufInteF2P_{BufferSize}.gdb'
    BufferInterFeature2PointPath = os.path.join(Sumup_Each_Landuse_NTL_Folder, BufferInterFeature2PointName)
    Check_GDBPath_Exist(BufferInterFeature2PointPath, Sumup_Each_Landuse_NTL_Folder, BufferInterFeature2PointName)
    # 执行Sumup_Each_Landuse_NTL函数
    # Process_4 = Sumup_Each_Landuse_NTL(ProcessGDBPath, BufferGDBPath, BufferIntersectPath,
    #                                    BufferInterFeature2PointPath, StartYear, EndYear, EulucPath)