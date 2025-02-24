import math

import numpy as np
import pandas as pd
from data_process.ProcessedData import ProcessedData


# 定义 Slice 类
class DynamicSliceData(ProcessedData):
    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = []

    def process(self, hasShape=False, mode="intersection"):
        if len(self.label_df) > 1:
            equal_zero_index = (self.label_df != 1).values
            equal_one_index = ~equal_zero_index

            fail_feature = np.array(self.feature_df[equal_one_index])

            ex_index = []  # 失败测试中的不执行的语句
            for temp in fail_feature:
                for i in range(len(temp)):
                    if temp[i] == 1:
                        ex_index.append(i)
            print("Slice result: ", list(set(ex_index)))

            select_index = sorted(list(set(ex_index)))
            print("Slice result: ", select_index, "\n", len(select_index))
            select_index = np.array(select_index)
            pca_index = self.process_pca()

            sliced_Len = len(select_index)
            inter_index = []
            if mode == "intersection":
                print("intersect Test")
                inter_index = np.intersect1d(select_index, pca_index[:sliced_Len])
            elif mode == "slice":
                print("Slice Test")
                inter_index = select_index
            elif mode == "pca":
                print("PCA Test")
                inter_index = pca_index[:sliced_Len]

            print("before Reshape: ", len(inter_index))
            req_shape = 4

            if hasShape:
                req_shape = 16

            print("req_shape: ", req_shape)
            if len(inter_index) % req_shape != 0 or len(inter_index) == 0:
                # slice_diff = np.setdiff1d(select_index, inter_index)
                slice_diff = np.array([x for x in select_index if x not in inter_index])
                # pca_diff = np.setdiff1d(pca_index, inter_index)
                pca_diff = np.array([x for x in pca_index if x not in inter_index])

                len_slice_diff = len(slice_diff)
                num_app = 0
                for temp in pca_diff:
                    if (len(inter_index) % req_shape == 0 and len(inter_index) != 0) or num_app == len_slice_diff:
                        break
                    if temp in slice_diff:
                        inter_index = np.append(inter_index, temp)
                        num_app += 1

                # pca_diff = np.setdiff1d(pca_diff, inter_index)
                pca_diff = np.array([x for x in pca_index if x not in inter_index])
                for temp in pca_diff:
                    if len(inter_index) % req_shape == 0 and len(inter_index) != 0:
                        break
                    inter_index = np.append(inter_index, temp)


            inter_index.sort()
            print("statements selected:", len(inter_index))
            print("statements selected:", self.feature_df.columns[inter_index])

            rest_index = [item for item in list(range(len(self.feature_df.iloc[0]))) if item not in inter_index]
            rest_columns = self.feature_df.columns[rest_index]
            self.rest_columns = list(rest_columns)
            low_features = self.feature_df.values.T[inter_index].T

            columns = self.feature_df.columns[inter_index]
            low_features = pd.DataFrame(low_features, columns=columns)
            low_data = pd.concat([low_features, self.label_df], axis=1)

            self.feature_df = low_features
            self.label_df = self.label_df
            self.data_df = low_data


            # print("Selected Indexes:", select_index)
            #
            # featureold = self.feature_df
            # sel_feature = self.feature_df.values.T[select_index].T
            # columns = self.feature_df.columns[select_index]
            # print("columns: ", columns)
            # self.feature_df = pd.DataFrame(sel_feature, columns=columns)
            # print("Processed Feature Data:")
            # print(self.feature_df)

    def process_pca(self, components_percent=0.7, eigenvalue_percent=0.7):
        # temp_len = self.feature_df.shape[1] * components_percent
        # temp_len = int(temp_len)
        # if temp_len < 8:
        #     return
        #
        # while (temp_len % 8 != 0.0):
        #     temp_len -= 1

        if len(self.label_df) > 1:
            covMatrix = self.feature_df.cov()  # 对CovMatrix求特征矩阵

            featValue, featVec = np.linalg.eig(covMatrix)
            print("featValue: ", featValue.shape, featVec.shape)
            index = np.argsort(-featValue)  # 返回 元素值递减 的角标
            eigenvalue_num = math.trunc(len(self.feature_df.values[0]) * eigenvalue_percent)  # 根据 percent 阶段数量
            selected_values = featValue[index[:eigenvalue_num]]
            selected_vectors = featVec.T[index[:eigenvalue_num]].T

            contri = np.array([sum(v) for v in np.abs(selected_vectors)])
            contri_index = np.argsort(-contri)

            return contri_index

        #     count = 0
        #     for i in contri_index:
        #         if i not in select_index:
        #             select_index.append(i)
        #             count = count + 1
        #             if (count == num_components):
        #                 break
        #
        # return sorted(select_index)

    def nearest_multiple_of_8(self, n):
        lower_multiple = (n // 8) * 8
        upper_multiple = ((n + 7) // 8) * 8
        if lower_multiple < 16:
            lower_multiple = 16
        if upper_multiple < 16:
            upper_multiple = 16

        if abs(n - lower_multiple) <= abs(n - upper_multiple):
            if lower_multiple <= 8:
                lower_multiple = 16
        else:
            if upper_multiple <= 8:
                upper_multiple = 16

        return lower_multiple, upper_multiple

# # 创建 Slice 实例并处理数据
# slice_instance = Slice(raw_data)
# slice_instance.process()
















# import math
#
# import numpy as np
# import pandas as pd
# from data_process.ProcessedData import ProcessedData
#
#
# # 定义 Slice 类
# class DynamicSliceData(ProcessedData):
#     def __init__(self, raw_data):
#         super().__init__(raw_data)
#         self.rest_columns = []
#
#     def process(self, hasShape=False):
#         if len(self.label_df) > 1:
#             equal_zero_index = (self.label_df != 1).values
#             equal_one_index = ~equal_zero_index
#
#             fail_feature = np.array(self.feature_df[equal_one_index])
#
#             ex_index = []  # 失败测试中的不执行的语句
#             for temp in fail_feature:
#                 for i in range(len(temp)):
#                     if temp[i] == 1:
#                         ex_index.append(i)
#
#             print(list(set(ex_index)))
#
#             select_index = sorted(list(set(ex_index)))
#             select_index = np.array(select_index)
#             pca_index = self.process_pca()
#
#             sliced_Len = len(select_index)
#
#             slice_matrix = np.array(self.feature_df.iloc[:, select_index])
#             pca_matrix = np.array(self.feature_df.iloc[:, pca_index])[:, :sliced_Len//4]
#
#             EPSILON = 1e-10
#             p = np.sum(slice_matrix == 1, axis=0) / slice_matrix.shape[0]
#             p = np.clip(p, EPSILON, 1 - EPSILON)
#             slice_matrix_entropy = - (p * np.log(p) + (1 - p) * np.log(1 - p))
#
#             p = np.sum(pca_matrix == 1, axis=0) / pca_matrix.shape[0]
#             p = np.clip(p, EPSILON, 1 - EPSILON)
#             pca_matrix_entropy = - (p * np.log(p) + (1 - p) * np.log(1 - p))
#             print("entropy", np.sum(slice_matrix_entropy), np.sum(pca_matrix_entropy))
#             if np.sum(slice_matrix_entropy) <= np.sum(pca_matrix_entropy):
#                 t_pca_index = []
#                 target = np.sum(slice_matrix_entropy)
#                 temp = 0
#                 args = np.argsort(pca_matrix_entropy)
#                 for i in args:
#                     if temp + pca_matrix_entropy[i] > target:
#                         break
#                     temp += pca_matrix_entropy[i]
#                     t_pca_index.append(pca_index[args[i]])
#                 inter_index = np.intersect1d(select_index, t_pca_index)
#             else:
#                 t_slice_index = []
#                 target = np.sum(pca_matrix_entropy)
#                 temp = 0
#                 args = np.argsort(slice_matrix_entropy)
#                 for i in args:
#                     if temp + slice_matrix_entropy[i] > target:
#                         break
#                     temp += slice_matrix_entropy[i]
#                     t_slice_index.append(select_index[args[i]])
#                 inter_index = np.intersect1d(t_slice_index, pca_index[:sliced_Len])
#             req_shape = 4
#
#             if hasShape:
#                 req_shape = 16
#
#             print("req_shape: ", req_shape)
#             if len(inter_index) % req_shape != 0 or len(inter_index) == 0:
#                 slice_diff = np.setdiff1d(select_index, inter_index)
#                 pca_diff = np.setdiff1d(pca_index, inter_index)
#
#                 len_slice_diff = len(slice_diff)
#                 num_app = 0
#                 for temp in pca_diff:
#                     if (len(inter_index) % req_shape == 0 and len(inter_index) != 0) or num_app == len_slice_diff:
#                         break
#                     if temp in slice_diff:
#                         inter_index = np.append(inter_index, temp)
#                         num_app += 1
#
#                 pca_diff = np.setdiff1d(pca_diff, inter_index)
#                 for temp in pca_diff:
#                     if len(inter_index) % req_shape == 0 and len(inter_index) != 0:
#                         break
#                     inter_index = np.append(inter_index, temp)
#
#
#             inter_index.sort()
#             print("statements selected:", len(inter_index))
#             print("statements selected:", self.feature_df.columns[inter_index])
#
#             rest_index = [item for item in list(range(len(self.feature_df.iloc[0]))) if item not in inter_index]
#             rest_columns = self.feature_df.columns[rest_index]
#             self.rest_columns = list(rest_columns)
#             low_features = self.feature_df.values.T[inter_index].T
#
#             columns = self.feature_df.columns[inter_index]
#             low_features = pd.DataFrame(low_features, columns=columns)
#             low_data = pd.concat([low_features, self.label_df], axis=1)
#
#             self.feature_df = low_features
#             self.label_df = self.label_df
#             self.data_df = low_data
#
#
#             # print("Selected Indexes:", select_index)
#             #
#             # featureold = self.feature_df
#             # sel_feature = self.feature_df.values.T[select_index].T
#             # columns = self.feature_df.columns[select_index]
#             # print("columns: ", columns)
#             # self.feature_df = pd.DataFrame(sel_feature, columns=columns)
#             # print("Processed Feature Data:")
#             # print(self.feature_df)
#
#     def process_pca(self, components_percent=0.7, eigenvalue_percent=0.7):
#         # temp_len = self.feature_df.shape[1] * components_percent
#         # temp_len = int(temp_len)
#         # if temp_len < 8:
#         #     return
#         #
#         # while (temp_len % 8 != 0.0):
#         #     temp_len -= 1
#
#         if len(self.label_df) > 1:
#             covMatrix = self.feature_df.cov()  # 对CovMatrix求特征矩阵
#
#             featValue, featVec = np.linalg.eig(covMatrix)
#             index = np.argsort(-featValue)  # 返回 元素值递减 的角标
#             eigenvalue_num = math.trunc(len(self.feature_df.values[0]) * eigenvalue_percent)  # 根据 percent 阶段数量
#             selected_values = featValue[index[:eigenvalue_num]]
#             selected_vectors = featVec.T[index[:eigenvalue_num]].T
#
#             contri = np.array([sum(v) for v in np.abs(selected_vectors)])
#             contri_index = np.argsort(-contri)
#
#             return contri_index
#
#         #     count = 0
#         #     for i in contri_index:
#         #         if i not in select_index:
#         #             select_index.append(i)
#         #             count = count + 1
#         #             if (count == num_components):
#         #                 break
#         #
#         # return sorted(select_index)
#
#     def nearest_multiple_of_8(self, n):
#         lower_multiple = (n // 8) * 8
#         upper_multiple = ((n + 7) // 8) * 8
#         if lower_multiple < 16:
#             lower_multiple = 16
#         if upper_multiple < 16:
#             upper_multiple = 16
#
#         if abs(n - lower_multiple) <= abs(n - upper_multiple):
#             if lower_multiple <= 8:
#                 lower_multiple = 16
#         else:
#             if upper_multiple <= 8:
#                 upper_multiple = 16
#
#         return lower_multiple, upper_multiple
#
# # # 创建 Slice 实例并处理数据
# # slice_instance = Slice(raw_data)
# # slice_instance.process()
