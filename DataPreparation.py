# -*- coding: UTF-8 -*-
import os
import time
import pandas as pd
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
    # 两个excel文件按照某一列合并，支持多线程版本
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


if __name__ == "__main__":
    outPath = r'E:/ShanghaiFactory/Shanghai_Final/'
    # 自定义Merge_Buf_BufInte_Excel函数的输入参数：
    BufferSize = '1500 METERS'  # <<Caution!!!>> 缓冲区的距离，这是一个可变参数，可选500m,1000m,1500m,2000m
    Sumup_Each_Landuse_NTL_Folder = os.path.join(outPath, 'Step02_Sumup_Each_Landuse_NTL/')
    OutInteExcelPath = os.path.join(Sumup_Each_Landuse_NTL_Folder, f'Sumup_Each_Landuse_NTL_BufInte_{BufferSize}')
    OutInteF2PExcelPath = os.path.join(Sumup_Each_Landuse_NTL_Folder, f'Sumup_Each_Landuse_NTL_BufInteF2P_{BufferSize}')
    Landuse_NTL_ExcelProcess_Folder = os.path.join(outPath, 'Step03_Landuse_NTL_ExcelProcess/')
    OutMergedBufInteExcelPath = os.path.join(Landuse_NTL_ExcelProcess_Folder, f'Landuse_NTL_MergedBufInte_{BufferSize}')
    Check_FolderPath_Exist(OutMergedBufInteExcelPath)
    Filenames = os.listdir(OutInteExcelPath)
    with ProcessPoolExecutor(max_workers=10) as executor:
        # 将 OutInteExcelPath, OutInteF2PExcelPath, OutMergedBufInteExcelPath 作为额外参数传递给 Merge_Buf_BufInte_Excel 函数
        executor.map(Merge_Buf_BufInte_Excel, [OutInteExcelPath] * len(Filenames),
                     [OutInteF2PExcelPath] * len(Filenames), [OutMergedBufInteExcelPath] * len(Filenames), Filenames)
