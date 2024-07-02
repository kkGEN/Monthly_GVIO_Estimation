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

def Sumup_Each_Landuse_NTL(RasGDBpath, InGDBPath, StartYear, EndYear, Industrial):
    # RasGDBpath：灯光数据数据库GDB
    # InGDBPath: 存储工厂缓冲区矢量范围的GDB
    # StartYear:
    # EndYear:
    # Industrial: Euluc土地利用矢量

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
                    print(str(BufferShp) + 'Inter')

                    # 为相交处理后的矢量图层增加主键
                    field_to_check = "BufferInterID"
                    field_exists = False
                    fields = arcpy.ListFields(BufferInterOutPath)
                    for field in fields:
                        if field.name == field_to_check:
                            field_exists = True
                            break
                    if not field_exists:
                        arcpy.AddField_management(BufferInterOutPath, field_to_check, 'LONG', 9, "", "", 'BufferInterID')
                        print(f"Field '{field_to_check}' added.")
                    else:
                        print(f"Field '{field_to_check}' already exists.")
                    arcpy.CalculateField_management(BufferInterOutPath, 'BufferInterID', expression, "PYTHON3", codeblock)

                    # 筛选面积大于3E-06的斑块，并输出为excel
                    arcpy.MakeFeatureLayer_management(BufferInterOutPath, "lyr")
                    arcpy.SelectLayerByAttribute_management("lyr", "NEW_SELECTION", "Shape_Area >= 3E-06")
                    arcpy.TableToExcel_conversion("lyr", BufferInterPath + str(BufferShp) + '.xlsx')
                    print(str(BufferShp) + ' Intersect Done!')

                    # 将添加主键后的矢量图层转换为点Points
                    BufferInterFeat2PointOutPath = BufferInterFeat2PointPath + str(BufferShp) + 'Inter_F2P'
                    # 将按面积筛选的BufferIntersect图层转换为点
                    arcpy.FeatureToPoint_management("lyr", BufferInterFeat2PointOutPath)
                    print(str(BufferShp) + ' Feature_to_Point Done!')

                    # 根据土地利用类型算后的斑块面积总和
                    LUNTL_temp_table = arcpy.CreateScratchName("SH_POItable", data_type="Table",
                                                               workspace="in_memory")
                    arcpy.sa.ZonalStatisticsAsTable(BufferInterFeat2PointOutPath, 'BufferInterID', rasSH,
                                                    LUNTL_temp_table)
                    arcpy.TableToExcel_conversion(LUNTL_temp_table, BufferInterZonalPath + str(BufferShp) + '.xlsx')
                    print(str(BufferShp) + ' Zonal_as_table Done!')
            # 切换回栅格图层所在数据库
            arcpy.env.workspace = RasGDBpath + gdbpath


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
    # Process_1 = Extract_Local_Monthly_DNB(OriginGlobalMonthlyDNB, AllGDBPath, StartYear, EndYear, ShapeFilePath)

    # 自定义Smooth_Monthly_DNB函数的输入参数：
    ProcessGDBPath = os.path.join(rootPath, 'SHMonthlyCompositionGDB_Preprocessed/')
    # 执行Smooth_Monthly_DNB函数
    # Process_2 = Smooth_Monthly_DNB(AllGDBPath, ProcessGDBPath)

    # 自定义Judge_ExtremeNTL函数的输入参数：
    Judge_ExtreNTL_Folder = os.path.join(outPath, 'Step01_Judge_ExtremeNTL/')
    ProjectFilePath = os.path.join(rootPath, 'sh_Local_Coordinate.prj')
    POIsIndustrialPath = os.path.join(rootPath, '上海基础空间数据/Shanghai_POI_Industrial.shp')

    BufferSize = '1500 METERS'  # <<Caution!!!>> 缓冲区的距离，这是一个可变参数，可选500m,1000m,1500m,2000m
    outExcelFolderPath = os.path.join(Judge_ExtreNTL_Folder, f'Judge_ExtremeNTL_{BufferSize}')  # 存放这个步骤的excel版本
    if not os.path.exists(outExcelFolderPath):
        os.makedirs(outExcelFolderPath)
    BufferGDBName = f'Judge_ExtremeNTL_Buffer_{BufferSize}.gdb'  # 存放该步骤得到的缓冲区结果，结果保留了缓冲区内是否存在极端值信息
    BufferGDBPath = os.path.join(Judge_ExtreNTL_Folder, BufferGDBName)  # 检查路径是否存在，不存在则新建
    if not os.path.exists(BufferGDBPath):
        try:
            arcpy.CreateFileGDB_management(Judge_ExtreNTL_Folder, BufferGDBName)
            print(f"{BufferGDBPath} is created successflly!")
        except arcpy.ExecuteError:
            # 打印错误信息
            print(arcpy.GetMessages(2))
    # 执行Judge_ExtremeNTL函数
    # Process_3 = Judge_Extreme_NTL(ProcessGDBPath, ProjectFilePath, outExcelFolderPath, BufferSize,
    #                              StartYear, EndYear, POIsIndustrialPath, BufferGDBPath)

    # 自定义Sumup_Each_Landuse_NTL函数的输入参数：
    EulucPath = os.path.join(rootPath, '上海基础空间数据/Shanghai_Euluc_2018.shp')
    # 以下步骤中的Judge_ExtreNTL_Folder路径来自于上一步Process_3的结果
    BufferIntersectGDBName = f'Judge_ExtremeNTL_Buffer_Intersect_{BufferSize}.gdb'
    BufferIntersectPath = os.path.join(Judge_ExtreNTL_Folder, BufferIntersectGDBName)
    BufferInterFeature2PointName = f'Judge_ExtremeNTL_Buf_Inte_Feature2Point_{BufferSize}.gdb'
    BufferInterFeature2PointPath = os.path.join(Judge_ExtreNTL_Folder, BufferInterFeature2PointName)
    Process_4 = Sumup_Each_Landuse_NTL(ProcessGDBPath, BufferGDBPath, StartYear, EndYear, EulucPath)







