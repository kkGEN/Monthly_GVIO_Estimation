# -*- coding: UTF-8 -*-
import os
import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def Check_FolderPath_Exist(outFolderPath):
    if not os.path.exists(outFolderPath):
        os.makedirs(outFolderPath)
        print(f'{outFolderPath} is created successflly!')
    return

def Timestamp():
    CurrentTime = time.strftime('%Y-%m-%d %H:%M:%S')
    return CurrentTime

def Merge_Buf_BufInte_Excel(folder1, folder2, output_folder, filename):
    # 用于两个excel文件按照某一列合并，支持多线程版本
    # folder1: 第一类excel的文件夹路径
    # folder2: 第二类excel的文件夹路径
    # output_folder: 输出excel的文件夹路径
    # filename: 每一个excel文件的名字，这里由于两类excel的命名是相同的，所以用同一个filename即可
    # 值得注意的是：这里的filename是多线程自主实现的每一个文件的循环，若不采用多线程，可在函数内部实现循环

    # 输出函数开始执行时间
    print("Start time is:{}".format(Timestamp()))

    # 构建完整的文件路径
    filepath1 = os.path.join(folder1, filename)
    filepath2 = os.path.join(folder2, filename)
    # 读取两个 Excel 文件
    df1 = pd.read_excel(filepath1)
    df2 = pd.read_excel(filepath2)
    # print(f"{filename} 文件读取成功!")
    # 指定连接的两列
    merge_columns = ['BufferInterID']
    # 使用 merge 函数连接两个表格
    merged_df = pd.merge(df1, df2, on=merge_columns)
    # print(f"{filename} 链接成功!")
    # 构建输出文件路径
    output_filepath = os.path.join(output_folder, filename)
    # 如果需要保存结果到新的 Excel 文件
    merged_df.to_excel(output_filepath, index=False)
    print(f"{filename} saved successfully!")

    # 输出函数结束时间
    print("End time is:{}".format(Timestamp()))
    return

def Sumup_NTL_Dependon_Landuse(mergedexcel_folder, NTLSum_folder, filename):
    # 用于将清晰和处理过的intersect数据按照原始ID恢复成一个，并添加了每一种土地利用类型下的灯光强度总和，
    # 最后按照卷积网络需要的格式reshape
    # mergedexcel_folder：需要处理的excel文件夹路径
    # NTLSum_folder：经过reshape后的土地利用灯光矩阵
    # filename：每一个excel文件的名字

    # 输出函数开始执行时间
    print("Start time is:{}".format(Timestamp()))

    # 构建完整的文件路径
    filepath = os.path.join(mergedexcel_folder, filename)
    df = pd.read_excel(filepath)
    # 提取ID的唯一值，并将对应的其他属性一起存储
    grouped_data = df.groupby('ID').agg({'MAX_x': 'first', 'MAXNTLCheck': 'first', 'BufferMAX': 'first'}).reset_index()
    # 根据Level2字段中的唯一值重新添加对应的列，并把同一ID下的多行SUM字段的各自求和分别填充到新添加的列中
    result = df.pivot_table(index='ID', columns='Level2', values='SUM', aggfunc='sum').reset_index()
    result.columns = [f'LU{col}NTLSUM' if col != 'ID' else col for col in result.columns]
    # 计算新添加的各列的总和，也就是不同土地利用类型下对应的灯光总和
    result['LUsumNTLSUM'] = result.iloc[:, 1:13].apply(lambda row: row.sum(), axis=1)
    result = result.fillna(0)
    # 将两个df按照ID链接形成按ID归类的不同土地利用类型下的灯光
    merged_df = pd.merge(grouped_data, result, on='ID')
    # 在不同的缓冲区下可能有的ID没有对应的值，出现缺失，为统一后期卷积操作的输入，将ID补全并赋值为0
    # 获取当前 ID 列的最小和最大值
    min_id, max_id = merged_df['ID'].min(), merged_df['ID'].max()
    # 创建包含完整连续整数范围的序列
    complete_id_range = pd.RangeIndex(start=min_id, stop=max_id + 1, step=1)
    # 使用 merge 将完整序列和原始 DataFrame 进行合并，填充缺失的 ID 行
    df_complete = complete_id_range.to_frame(name='ID').merge(merged_df, on='ID', how='left').fillna(0)
    df_complete.to_excel(os.path.join(NTLSum_folder, filename))

    # 输出函数结束时间
    print("End time is:{}".format(Timestamp()))
    return

def Find_MaxNTl_and_Greater20_Landuse(CleanedExcelPath, OutExcelpath):
    # 读取不同土地利用类型下灯光总和并判断各工厂灯光水平是否超过阈值（20），及最高灯光总和对应的土地利用类型
    # CleanedExcelPath: 经过reshape后的土地利用灯光矩阵
    # OutExcelpath: 增加超过阈值判断，最强灯光对用土地利用类型的结果

    # 输出函数开始执行时间
    print("Start time is:{}".format(Timestamp()))

    files = os.listdir(CleanedExcelPath)
    for file in files:
        # 判断是否超过阈值
        df_all = pd.DataFrame(columns=['Max>20'])
        df_file = pd.read_excel(os.path.join(CleanedExcelPath, file))
        df_temp = df_file.iloc[:, 2:16]
        df_all['Max>20'] = df_temp['MAX_x'].apply(lambda x: 1 if x > 20 else 0)
        df_all = pd.concat([df_all, df_temp], axis=1)

        # 判断最大灯光总和对应土地利用类型
        df_findmax = df_all
        # 找到每行中的最大值并记录到max列中
        df_findmax['max'] = df_findmax.max(axis=1)
        # 记录最大值所在列的列名
        df_findmax['max_col'] = df_findmax.idxmax(axis=1)
        # 使用mapping方法将土地利用类型重新编号为0-11
        mapping = {'': 0,'LU101NTLSUM': 1, 'LU201NTLSUM': 2, 'LU202NTLSUM': 3, 'LU301NTLSUM': 4, 'LU402NTLSUM': 5,
            'LU403NTLSUM': 6, 'LU501NTLSUM': 7, 'LU502NTLSUM': 8, 'LU503NTLSUM': 9, 'LU504NTLSUM': 10, 'LU505NTLSUM': 11}
        df_findmax['MaxLuNtlCheck'] = df_findmax['max_col'].map(mapping)
        df_findmax = df_findmax.drop(labels=['max', "max_col"], axis=1).fillna(0)
        df_findmax.to_excel(os.path.join(OutExcelpath, f"MaxLuNtl_{file}"))
        print(f"{file} has been processed successfully!")

    # 输出函数结束时间
    print("End time is:{}".format(Timestamp()))
    return

def Shift_Data_to_CNN_Shape(folderpath, outfilepath):
    # 把原始的长条数据reshape变为方便CNN处理的方形数据
    # folderpath：经过地理处理后的各土地利用类型下的灯光强度矩阵
    # outfilepath：满足CNN输入数据的输出路径

    # 输出函数开始执行时间
    print("Start time is:{}".format(Timestamp()))

    files = os.listdir(folderpath)
    for filename in files:
        df_all = pd.read_excel(os.path.join(folderpath, filename))
        df_all = df_all.iloc[:, 1:17]
        # 计算原始df的列数和大小，存储列名
        df_cols = df_all.shape[1]
        df_size = df_all.size
        df_colnames = df_all.columns
        # 计算新的方形矩阵的边长（向上取整）
        side_length = int(np.ceil(np.sqrt(df_size)))
        if side_length % 2 != 0:
            side_length += 1
        # 计算需要填充的零的数量
        num_rows_to_add = int((side_length * side_length - df_size) / df_cols)
        zero_rows = pd.DataFrame(np.zeros((num_rows_to_add, df_cols)), columns=df_colnames)
        # 在数组的末尾添加零
        df_all_with_zeros = pd.concat([df_all, zero_rows], ignore_index=True)
        array_data = df_all_with_zeros.values
        # 重塑数组为方形
        reshaped_array = array_data.reshape((side_length, side_length))

        # #填补空缺和零值为0.001
        reshaped_df = pd.DataFrame(reshaped_array)
        reshaped_df_out = reshaped_df.replace(0, 0.0001)
        reshaped_df_out.to_excel(os.path.join(outfilepath, filename), header=None, index=False)
        print(f"{filename} has been processed successfully!")

    # 输出函数结束时间
    print("End time is:{}".format(Timestamp()))
    return

if __name__ == "__main__":
    outPath = r'E:/ShanghaiFactory/Shanghai_Final/'
    # 自定义Merge_Buf_BufInte_Excel函数的输入参数：
    BufferSize = '1000 METERS'  # <<Caution!!!>> 缓冲区的距离，这是一个可变参数，可选500m,1000m,1500m,2000m
    Sumup_Each_Landuse_NTL_Folder = os.path.join(outPath, 'Step02_Sumup_Each_Landuse_NTL/')
    OutInteExcelPath = os.path.join(Sumup_Each_Landuse_NTL_Folder, f'Sumup_Each_Landuse_NTL_BufInte_{BufferSize}')
    OutInteF2PExcelPath = os.path.join(Sumup_Each_Landuse_NTL_Folder, f'Sumup_Each_Landuse_NTL_BufInteF2P_{BufferSize}')
    Landuse_NTL_ExcelProcess_Folder = os.path.join(outPath, 'Step03_Landuse_NTL_ExcelProcess/')
    OutMergedBufInteExcelPath = os.path.join(Landuse_NTL_ExcelProcess_Folder, f'Landuse_NTL_MergedBufInte_{BufferSize}')
    Check_FolderPath_Exist(OutMergedBufInteExcelPath)
    # 列出文件夹中的所有文件名
    Filenames = os.listdir(OutInteExcelPath)
    # 自定义线程的数量
    MaxWorks = 10
    # Process_5
    with ProcessPoolExecutor(max_workers=MaxWorks) as executor:
        # 将 OutInteExcelPath, OutInteF2PExcelPath, OutMergedBufInteExcelPath 作为额外参数传递给 Merge_Buf_BufInte_Excel 函数
        executor.map(Merge_Buf_BufInte_Excel, [OutInteExcelPath] * len(Filenames),
                     [OutInteF2PExcelPath] * len(Filenames), [OutMergedBufInteExcelPath] * len(Filenames), Filenames)

    # 自定义Sumup_NTL_Dependon_Landuse函数的输入参数：
    ProcessFiles = os.listdir(OutMergedBufInteExcelPath)
    OutCleanedExcelPath = os.path.join(Landuse_NTL_ExcelProcess_Folder, f'Landuse_NTL_Sumup_Cleaned_{BufferSize}')
    Check_FolderPath_Exist(OutCleanedExcelPath)
    # Process_6
    with ProcessPoolExecutor(max_workers=MaxWorks) as executor:
        executor.map(Sumup_NTL_Dependon_Landuse, [OutMergedBufInteExcelPath] * len(ProcessFiles),
                     [OutCleanedExcelPath] * len(ProcessFiles), ProcessFiles)

    # 自定义Find_MaxNTl_and_Greater20_Landuse函数的输入参数：
    OutCheckedMaxExcelPath = os.path.join(Landuse_NTL_ExcelProcess_Folder, f'Landuse_NTL_Sumup_ChecekedMax_{BufferSize}')
    Check_FolderPath_Exist(OutCheckedMaxExcelPath)
    # 执行Find_MaxNTl_and_Greater20_Landuse函数
    Process_7 = Find_MaxNTl_and_Greater20_Landuse(OutCleanedExcelPath, OutCheckedMaxExcelPath)

    # 自定义Shift_Data_to_CNN_Shape函数的输入参数：
    OutShiftDataCNNExcelPath = os.path.join(Landuse_NTL_ExcelProcess_Folder, f'Landuse_NTL_CNNdata_{BufferSize}')
    Check_FolderPath_Exist(OutShiftDataCNNExcelPath)
    Process_8 = Shift_Data_to_CNN_Shape(OutCheckedMaxExcelPath, OutShiftDataCNNExcelPath)

