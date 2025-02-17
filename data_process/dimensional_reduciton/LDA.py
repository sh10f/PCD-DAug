import math

import numpy as np
import pandas as pd
from data_process.ProcessedData import ProcessedData
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class LDAData(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = None

    def process(self, components_percent=0.7, eigenvalue_percent=0.7):
        if len(self.label_df) > 1:
            Sw = np.zeros((self.feature_df.shape[1], self.feature_df.shape[1]))
            for i in range(2):
                datai = self.feature_df[(self.label_df == i).values]
                datai = datai - datai.mean(0)
                #Swi = np.mat(datai).T * np.mat(datai)
                Swi=np.mat(datai).T.dot(np.mat(datai))
                Sw += Swi

            SB = np.zeros((self.feature_df.shape[1], self.feature_df.shape[1]))
            u = self.feature_df.mean(0) 
            for i in range(2):
                Ni = self.feature_df[(self.label_df == i).values].shape[0]
                ui = self.feature_df[(self.label_df == i).values].mean(0)
                #SBi = Ni * np.mat(ui - u).T * np.mat(ui - u)
                SBi = Ni * np.mat(ui - u).T.dot( np.mat(ui - u))
                SB += SBi

            S=SB-Sw
            featValue, featVec = np.linalg.eig(S)  
            index = np.argsort(-featValue)

            eigenvalue_num = math.trunc(len(self.feature_df.values[0]) * eigenvalue_percent)
            selected_values = featValue[index[:eigenvalue_num]]
            selected_vectors = featVec.T[index[:eigenvalue_num]].T

            contri = np.array([sum(v) for v in np.abs(selected_vectors)])
            contri_index = np.argsort(-contri)

            num_components = math.trunc(len(self.feature_df.values[0]) * components_percent)
            selected_index = contri_index[:num_components]
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
