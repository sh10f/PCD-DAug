import pandas as pd
import numpy as np
from data_process.ProcessedData import ProcessedData

# 原始数据
data = {
    10000: [0, 0, 0, 1, 1],
    10002: [1, 0, 1, 1, 0],
    20000: [0, 1, 0, 1, 0],
    25000: [1, 1, 1, 0, 0],
    30000: [0, 1, 1, 0, 1],
    'label': [1, 1, 0, 0, 0]
}

raw_data = pd.DataFrame(data)

# 新的列
new_column = {20006: [1, 1, 1, 0, 0]}

# 将新的列添加到 DataFrame 中
raw_data[20006] = new_column[20006]

# 重新排序列
sorted_columns = sorted(raw_data.columns, key=lambda x: (isinstance(x, int), x))
raw_data = raw_data[sorted_columns]

# 输出结果
print(raw_data)





# 模拟 ProcessedData 类
class ProcessedData:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.feature_df = raw_data.drop(columns=['label'])
        self.label_df = raw_data['label']

# 定义 Slice 类
class Slice(ProcessedData):
    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = raw_data.columns.tolist()

    def process(self):
        if len(self.label_df) > 1:
            equal_zero_index = (self.label_df != 1).values
            equal_one_index = ~equal_zero_index

            fail_feature = np.array(self.feature_df[equal_one_index])

            ex_index = []  # 失败测试中的不执行的语句
            for temp in fail_feature:
                for i in range(len(temp)):
                    if temp[i] == 1:
                        ex_index.append(i)

            print(list(set(ex_index)))

            select_index = list(set(ex_index))

            # select_index = []  # 所有失败测试用例中都执行的语句
            # for i in range(len(self.feature_df.values[0])):
            #     if i not in ex_index:
            #         select_index.append(i)
            #
            # ex_index = list(set(ex_index))
            # select_index = list(set(select_index))

            # 在这里选择特定的索引进行演示
            # select_index = [0, 2, 4]
            print("Selected Indexes:", select_index)

            featureold = self.feature_df
            sel_feature = self.feature_df.values.T[select_index].T
            columns = self.feature_df.columns[select_index]
            print("columns: ", columns)
            self.feature_df = pd.DataFrame(sel_feature, columns=columns)
            print("Processed Feature Data:")
            print(self.feature_df)

# 创建 Slice 实例并处理数据
slice_instance = Slice(raw_data)
slice_instance.process()
