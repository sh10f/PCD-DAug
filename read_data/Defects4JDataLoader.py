import os
import re
import pandas as pd
import sys
from utils.args_util import parse_args
from utils.file_util import *
from read_data.DataLoader import DataLoader

# 重点在于
# self.feature_df--CovMatrix
# label_df---error vector
# data_df---CovMatrix + Error Vector
# concrete_columns---对应列号列表
# fault_line---错误statement的列号列表
class Defects4JDataLoader(DataLoader):

    def __init__(self, base_dir, program, bug_id):
        # base_dir---xxxx/data  program---d4j   bug_id---1
        super().__init__(base_dir, program, bug_id)

    def load(self):
        self.file_dir = os.path.join(self.base_dir,
                                     "d4j",
                                     "data",
                                     self.program, str(self.bug_id),
                                     "gzoltars",
                                     self.program, str(self.bug_id)) # 再往下就是 log.txt,matrix.txt 那一层了
        self._load_columns()
        self._load_features()
        self._load_fault_line()

    def _load_features(self):
        self.matrix_path = os.path.join(self.file_dir, 'matrix')
        feature_data = process_coding(self.matrix_path) # 返回经过相应编码解码后的文件内容
        feature_data, label_data = self._process_feature_data(feature_data) # 此处已经为 0-1矩阵
        self.feature_df = pd.DataFrame(feature_data, columns=self.concrete_column[:])  # concrete_column 表示列号
        self.label_df = pd.DataFrame(label_data, columns=['error'])
        self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)

    def _load_columns(self):
        columns_path = os.path.join(self.file_dir, 'spectra')
        concrete_columns = self._process_content(columns_path) # 返回一个列表
        # self.concrete_column---列号  self.columnmap---类名列表
        self.concrete_column, self.columnmap = self._getnewcolumns(concrete_columns)

    def _load_fault_line(self):
        fault_dir = os.path.join(self.base_dir,"d4j", "buggy-lines", self.program + "-" + str(self.bug_id) + ".buggy.lines")

        fault_line_data = process_coding(fault_dir)
        fault_line_data = self._process_fault_line_data(fault_line_data)

        # 返回 fault_line 列号的列表--eg: 16850
        self.fault_line = [self._cal_column(i, self.columnmap) for i in fault_line_data]

    def _getnewcolumns(self, classnames):
        names = []
        for i in classnames:
            name = re.sub('#.*', '', str(i))  # 使用正则表达式去除类名中的 # 及其后面的内容，得到纯净的类名
            if name not in names:
                names.append(str(name))
        columns = []
        for i in classnames:
            columns.append(int(self._cal_column(str(i), names)))

        # colunms---列号   names---类名列表
        return columns, names

    def _cal_column(self, s, data):
        # s--org.jfree.chart.util.DefaultShadowGenerator#321 data---[org.jfree.chart.util.DefaultShadowGenerator, xxxxx, xxx]
        s = str(s)
        classname = re.sub('#.*', '', s)
        num = int(re.sub('.*#', '', s))     # 返回对应的数字
        classnum = data.index(classname)
        column = (classnum + 1) * 100000+ num
        return int(column)

    def _process_fault_line_data(self, fault_line_data):
        temp_data = re.findall(".*#\d+", fault_line_data)
        temp_data = [i.replace(r'.java', '') for i in temp_data]
        temp_data = [i.replace('/', '.') for i in temp_data]
        temp_data = [i.strip() for i in temp_data]
        return list(map(str, temp_data))

    def _process_content(self, columns_path):
        columns = process_coding(columns_path)
        token = choose_newlines(columns)

        concrete_columns = columns.split(token)

        return concrete_columns

    def _process_feature_data(self, feature_data):
        token = choose_newlines(feature_data) # 确定当前操作系统采用的换行符是 “\r\n" or "\n" or "\r"
        feature_data = feature_data.split(token)

        feature_data = [feature_str.strip().split() for feature_str in feature_data] # 包含label

        label_data = [arr[-1] for arr in feature_data]

        label_data = [0 if a == '+' else 1 for a in label_data]

        feature_data = [arr[:-1] for arr in feature_data]  # 去除最后的label
        feature_data = [list(map(int, arr)) for arr in feature_data] # 转换为int型元素

        return feature_data, label_data


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(__file__))
    project_dir += "\\data"
    print(project_dir)
    data_loader = Defects4JDataLoader(project_dir, "Chart", 1)
    data_loader.load()
    print(data_loader.data_df.shape)
    print(data_loader.feature_df.shape)
    print(data_loader.label_df.shape)
    data_loader.data_df.to_csv("data.csv")

    print(data_loader.feature_df[:10])
    print("********************")

    print(data_loader.label_df[:10])

    rows_with_1 = data_loader.label_df[data_loader.label_df['error'] == 1]
    print(rows_with_1)

