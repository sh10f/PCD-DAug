import math

import numpy as np
import pandas as pd
from data_process.ProcessedData import ProcessedData


class PCAData(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        # raw_data 是一个 dataloader，含有self.feature_df等等
        self.rest_columns = None

    def process(self, components_percent=0.7, eigenvalue_percent=0.7):
        temp_len = self.feature_df.shape[1] * components_percent
        temp_len = int(temp_len)
        if temp_len < 8:
            return

        while (temp_len % 8 != 0.0):
            temp_len -= 1

        if len(self.label_df) > 1:
            covMatrix = self.feature_df.cov()  # 对CovMatrix求特征矩阵

            featValue, featVec = np.linalg.eig(covMatrix)
            index = np.argsort(-featValue)  # 返回 元素值递减 的角标
            eigenvalue_num = math.trunc(len(self.feature_df.values[0]) * eigenvalue_percent)  # 根据 percent 阶段数量
            selected_values = featValue[index[:eigenvalue_num]]
            selected_vectors = featVec.T[index[:eigenvalue_num]].T

            contri = np.array([sum(v) for v in np.abs(selected_vectors)])
            contri_index = np.argsort(-contri)

            # num_components = math.trunc(len(self.feature_df.values[0]) * components_percent)
            num_components = temp_len
            selected_index = contri_index[:num_components]  # 选择的 statements 角标
            rest_index = contri_index[num_components:]
            rest_columns = self.feature_df.columns[rest_index]
            self.rest_columns = list(rest_columns)
            low_features = self.feature_df.values.T[selected_index].T

            columns = self.feature_df.columns[selected_index]
            low_features = pd.DataFrame(low_features, columns=columns)
            low_data = pd.concat([low_features, self.label_df], axis=1)

            self.feature_df = low_features
            self.label_df = self.label_df
            self.data_df = low_data
