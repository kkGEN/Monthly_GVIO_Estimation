# -*- coding: UTF-8 -*-
import arcpy
import os
import time
import arcpy.da
arcpy.CheckOutExtension("spatial")  # 权限检查
arcpy.env.overwriteOutput = True

def Timestamp():
    CurrentTime = time.strftime('%Y-%m-%d %H:%M:%S')
    return CurrentTime

def Judge_Extreme_NTL(inGDBpath, outExcelPath, Buffersize, StartYear, EndYear, POI, OutshpPath):
    # 判断每个工厂的缓冲区内是否存在极端大的灯光值
    # inGDBpath: 要处理的原始数据的GDB数据库路径集合
    # outExcelPath: 每一个缓冲区距离下的输出excel存储文件夹路径
    # Buffersize：自定义缓冲区距离
    # POI: 输入中类为”工厂“的POI
    # OutshpPath: 输出工厂POI的缓冲区，并存储了是否存在极端灯光值信息
    # StartYear：需处理数据的起始年份
    # EndYear：需处理数据的结束年份

    # 输出函数开始执行时间
    print("Start time is: {}".format(Timestamp()))
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
            # POIbufferRows.clear()
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
    # 输出函数结束时间
    print("End time is: {}".format(Timestamp()))
    return

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
        print(f'{outFolderPath} is created successfully!')
    return

def Sumup_Each_Landuse_NTL(RasGDBpath, InGDBPath, BuffInterPath, BuffInterF2PPath,
                           BuffInterExcelPath,BuffInterF2PExcelPath, StartYear, EndYear, Industrial):
    # RasGDBpath：灯光数据数据库GDB
    # InGDBPath: 存储工厂缓冲区矢量范围的GDB
    # StartYear:
    # EndYear:
    # Industrial: Euluc土地利用矢量

    # 输出函数开始执行时间
    print("Start time is:{}".format(Timestamp()))
    # 循环打开每年的月合成灯光数据库
    for year in range(StartYear, EndYear):
        GdbPath = f"{year}.gdb"
        OriginalRasGDB = os.path.join(RasGDBpath, GdbPath)
        # 确保初始数据库是在栅格灯光数据中
        arcpy.env.workspace = OriginalRasGDB
        # 打开每月的灯光栅格影像
        for ras in arcpy.ListRasters():
            Ras = arcpy.Raster(ras)
            # 切换到工厂缓冲区矢量数据库，列出带缓冲区的工厂POI图层
            arcpy.env.workspace = InGDBPath
            arcpy.env.overwriteOutput = True
            BufferShpList = arcpy.ListFeatureClasses()
            for BufferShp in BufferShpList:
                if f'{Ras}' in BufferShp:
                    # 输出buffer与EULUC相交的结果
                    BufferInterOutPath = os.path.join(BuffInterPath, f'{BufferShp}_Intersect')
                    if arcpy.Exists(BufferInterOutPath):
                        print(f'{BufferShp}_inter exists!')
                    else:
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
                        print(f"Field '{field_to_check}' is not exist, added now!")
                    else:
                        print(f"Field '{field_to_check}' already exists!")
                    # 为新增的BufferInterID字段进行赋值，按照升序依次编号
                    IDCount = 1
                    IntersectRows = arcpy.UpdateCursor(BufferInterOutPath, [field_to_check])
                    while True:
                        interrow = IntersectRows.next()
                        if not interrow:
                            break
                        interrow.setValue(field_to_check, IDCount)
                        IntersectRows.updateRow(interrow)
                        IDCount += 1
                    print(f"{BufferShp} attribute: '{field_to_check}' assigned successfully!")
                    # <<with 方法在处理游标之类需要处理内存的数据类型时更优，但是目前报错：AttributeError: __enter__，
                    # 通常在 ArcPy 中，arcpy.da.UpdateCursor 和 arcpy.da.SearchCursor 都是可以用于 with 语句的上下文管理器，
                    # 可能由于arcpy版本不适配，或者arcpy.da没有正确导入>>
                    # with arcpy.UpdateCursor(BufferInterOutPath, [f'{field_to_check}']) as cursor:
                    #     for row in cursor:
                    #         # 按照递增的顺序设置新字段的值
                    #         row.setValue(field_to_check, IDCount)
                    #         cursor.updateRow(row)
                    #         IDCount += 1

                    # 筛选面积大于3E-06的斑块，并输出为excel
                    # 复制一个当前BufferInter图层的内存版本，只保留部分字段，且面积大于或等于 0.000003 平方单位的要素
                    # SelectedBufInte = "SelectedBufferInter"
                    WhereClauseArea = "Shape_Area >= 3E-06"
                    SecletedFiled = ['ID', 'BUFF_DIST', 'MAX', 'MAXNTLCheck', 'BufferMAX',
                                     'Lon', 'Lat', 'Level1', 'Level2', 'Shape_Area', 'BufferInterID']
                    FieldInfo = arcpy.FieldInfo()
                    FieldsInBuffInte = arcpy.ListFields(BufferInterOutPath)
                    for fieldbuffinte in FieldsInBuffInte:
                        if fieldbuffinte.name in SecletedFiled:
                            FieldInfo.addField(fieldbuffinte.name, fieldbuffinte.name, "VISIBLE", "NONE")
                        else:
                            FieldInfo.addField(fieldbuffinte.name, fieldbuffinte.name, "HIDDEN", "NONE")
                    arcpy.MakeFeatureLayer_management(BufferInterOutPath, "SelectedBufferInter", where_clause=WhereClauseArea, field_info=FieldInfo)

                    outInterExcelName = os.path.join(BuffInterExcelPath, f'{BufferShp}.xlsx')
                    arcpy.TableToExcel_conversion("SelectedBufferInter", outInterExcelName)
                    print(f'{BufferShp} Intersect Done!')

                    # 将添加主键后的矢量，并且按面积筛选的BufferIntersect图层转换为点（Points）
                    BuffInterF2POutPath = os.path.join(BuffInterF2PPath, f'{BufferShp}Inter_F2P')
                    arcpy.FeatureToPoint_management("SelectedBufferInter", BuffInterF2POutPath)
                    print(f'{BufferShp} Feature_to_Point Done!')

                    # (临时文件)根据土地利用类型算后的斑块面积总和
                    LUNTL_Temp_table = arcpy.CreateScratchName("SH_POItable", data_type="Table", workspace="in_memory")
                    OrigRas =arcpy.Raster( os.path.join(OriginalRasGDB, ras))
                    arcpy.sa.ZonalStatisticsAsTable(BuffInterF2POutPath, 'BufferInterID', OrigRas, LUNTL_Temp_table)
                    outInterF2PExcelName = os.path.join(BuffInterF2PExcelPath, f'{BufferShp}.xlsx')
                    arcpy.TableToExcel_conversion(LUNTL_Temp_table, outInterF2PExcelName)
                    arcpy.Delete_management(LUNTL_Temp_table)
                    print(f'{BufferShp} Zonal_as_table Done!')
            # 切换回栅格图层所在数据库
            arcpy.env.workspace = OriginalRasGDB

    # 输出函数结束时间
    print("End time is:{}".format(Timestamp()))
    return


if __name__ == '__main__':
    rootPath = r'E:/ShanghaiFactory/'
    outPath = r'E:/ShanghaiFactory/Shanghai_Final/'
    StartYear = 2014
    EndYear = 2023

    BufferSize = '1500 METERS'  # <<Caution!!!>> 缓冲区的距离，这是一个可变参数，可选500m,1000m,1500m,2000m

    # 自定义Judge_ExtremeNTL函数的输入参数：
    Judge_ExtreNTL_Folder = os.path.join(outPath, 'Step01_Judge_Extreme_NTL/')
    POIsIndustrialPath = os.path.join(rootPath, 'SHBasicGeoData/Shanghai_POI_Industrial.shp')
    ProcessGDBPath = os.path.join(rootPath, 'SHMonthlyCompositionGDB_Preprocessed/')

    # 存放Buffer之后的excel版本
    outExcelFolderPath = os.path.join(Judge_ExtreNTL_Folder, f'Judge_Extreme_NTL_{BufferSize}')
    Check_FolderPath_Exist(outExcelFolderPath)
    BufferGDBName = f'Judge_Extreme_NTL_Buffer_{BufferSize}.gdb'  # 存放该步骤得到的缓冲区结果，结果保留了缓冲区内是否存在极端值信息
    BufferGDBPath = os.path.join(Judge_ExtreNTL_Folder, BufferGDBName)
    Check_GDBPath_Exist(BufferGDBPath, Judge_ExtreNTL_Folder, BufferGDBName)  # 检查路径是否存在，不存在则新建

    # 执行Judge_Extreme_NTL函数
    Process_3 = Judge_Extreme_NTL(ProcessGDBPath, outExcelFolderPath, BufferSize,
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
    outBufferIntersectPath = os.path.join(Sumup_Each_Landuse_NTL_Folder, BufferIntersectGDBName)
    Check_GDBPath_Exist(outBufferIntersectPath, Sumup_Each_Landuse_NTL_Folder, BufferIntersectGDBName)
    # 存放Intersect后并FeatureToPoints的GDB路径
    BufferInterFeature2PointName = f'Sumup_Each_Landuse_NTL_BufInteF2P_{BufferSize}.gdb'
    outBufferInterFeature2PointPath = os.path.join(Sumup_Each_Landuse_NTL_Folder, BufferInterFeature2PointName)
    Check_GDBPath_Exist(outBufferInterFeature2PointPath, Sumup_Each_Landuse_NTL_Folder, BufferInterFeature2PointName)
    # 执行Sumup_Each_Landuse_NTL函数
    Process_4 = Sumup_Each_Landuse_NTL(ProcessGDBPath, BufferGDBPath, outBufferIntersectPath,
                                       outBufferInterFeature2PointPath, OutInteExcelPath, OutInteF2PExcelPath,
                                       StartYear, EndYear, EulucPath)