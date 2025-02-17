import os
import re

import pandas as pd

from utils.file_util import *
from read_data.DataLoader import DataLoader

# 重点在于
# self.feature_df--CovMatrix
# label_df---error vector
# data_df---CovMatrix + Error Vector
# concrete_columns---对应列号列表
# fault_line---错误statement的列号列表
class SlicedDataLoader(DataLoader):
    def __init__(self, base_dir, program, bug_id):
        super().__init__(base_dir, program, bug_id)

    def load(self):
        self.file_dir = os.path.join(self.base_dir,
                                     self.program,
                                     str(self.bug_id)
                                     )
        self._load_columns()
        self._load_features()
        self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)
        self._load_fault_line()

    def _load_features(self):
        feature_path = os.path.join(self.file_dir, 'cc_sample.csv')
        fault_path = os.path.join(self.file_dir, 'fault_sample.csv')
        preview_df = pd.read_csv(feature_path, nrows=5)
        # 确定总列数
        num_columns = len(preview_df.columns)
        columns = list(range(1, num_columns - 1))

        # 使用 read_csv 函数读取 CSV 文件，通过 usecols 参数指定要读取的列范围
        df = pd.read_csv(feature_path, usecols=columns)
        df_fault = pd.read_csv(fault_path, usecols=columns)

        df = pd.concat([df, df_fault], axis=0)
        # 显示读取的数据框
        df.columns = self.concrete_columns[:]
        self.feature_df = df

        self._load_labels()

    def _load_labels(self):
        feature_path = os.path.join(self.file_dir, 'cc_sample.csv')
        fault_path = os.path.join(self.file_dir, 'fault_sample.csv')
        preview_df = pd.read_csv(feature_path, nrows=5)
        # 确定总列数
        num_columns = len(preview_df.columns)
        columns = [num_columns - 1]

        # 使用 read_csv 函数读取 CSV 文件，通过 usecols 参数指定要读取的列范围
        df = pd.read_csv(feature_path, usecols=columns)
        df_fault = pd.read_csv(fault_path, usecols=columns)

        df = pd.concat([df, df_fault], axis=0)
        # 显示读取的数据框
        df.columns = ['error']
        self.label_df = df

    def _load_columns(self):
        columns_path = os.path.join(self.file_dir, 'cc_sample.csv')
        self.concrete_columns = self._process_content(columns_path)

    def _load_fault_line(self):
        fault_line_data = process_coding(os.path.join(self.file_dir, "bugline.txt"))
        print(fault_line_data)
        self.fault_line = fault_line_data
        # self.fault_line = self._process_fault_line_data(fault_line_data)


    def _process_fault_line_data(self, fault_line_data):
        temp_data = re.findall("\"(.*?)\"", fault_line_data)[0]
        temp_data = temp_data.strip().split()
        return list(map(int, temp_data))

    def _process_label_data(self, label_data):
        token = choose_newlines(label_data)
        label_data = label_data.split(token)

        label_data = [list(map(int, arr)) for arr in label_data]
        return label_data

    def _process_content(self, columns_path):
        import csv
        with open(columns_path, 'r', newline='') as file:
            reader = csv.reader(file)
            first_line = next(reader)

        first_line = first_line[1:]
        first_line = first_line[:-1]
        concrete_columns = list(map(int, first_line))
        return concrete_columns

    def _process_feature_data(self, feature_data):

        token = choose_newlines(feature_data)
        feature_data = feature_data.split(token)

        feature_data = [feature_str.strip().split() for feature_str in feature_data]
        feature_data = [list(map(int, arr)) for arr in feature_data]
        feature_data = [[0 if a == 0 else 1 for a in elem] for elem in feature_data]

        return feature_data


if __name__ == '__main__':
    project_dir = r"D:\university_study\科研\slice\contextdata\ContextData\Fixed_data_v1"
    program = "Chart"
    bug_id = 1
    loader = SlicedDataLoader(project_dir, program, bug_id)
    loader.load()
    print(len(loader.concrete_columns))
    print(loader.feature_df)
    print("*************")
    print(loader.label_df)
    print(len(loader.label_df))
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print(loader.data_df)

    print(loader.fault_line)
