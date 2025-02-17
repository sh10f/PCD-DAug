import os
from read_data.ManyBugsDataLoader import ManyBugsDataLoader
from read_data.Defects4JDataLoader import Defects4JDataLoader
from read_data.SIRDataLoader import SIRDataLoader
from read_data.SlicedDataLoader import SlicedDataLoader
from read_data.NewD4jDataLoader import NewD4jDataLoader

from data_process.data_systhesis.resampling import ResamplingData
from data_process.data_systhesis.smote import SMOTEData
from data_process.data_systhesis.cvae_synthesis import CVAESynthesisData
from data_process.data_systhesis.diffusion_synthesis import ConDiffusionSynData
from data_process.data_systhesis.mixup import MixUp
from data_process.data_systhesis.CGAN import CGANSynthesisData

from data_process.dimensional_reduciton.LDA import LDAData
from data_process.dimensional_reduciton.Slice import Slice
from data_process.dimensional_reduciton.PCA import PCAData
from data_process.dimensional_reduciton.DynamicSlice import DynamicSliceData
from data_process.data_undersampling.undersampling import UndersamplingData
from calculate_suspiciousness.CalculateSuspiciousness import CalculateSuspiciousness


class Pipeline:
    def __init__(self, project_dir, configs):
        """
        :param project_dir:
        :param configs: 字典————指示了实验采用的相关方法
        """
        self.configs = configs
        self.project_dir = project_dir

        # 直接给dataset的相对位置 -- d4j所在的文件夹
        # self.project_dir = r"D:\university_study\科研\Diffusion_Model"
        self.project_dir = "/home/zhangxiaohong/yangjunzhe/temp_F/datasets/"

        self.dataset = configs["-d"]
        self.program = configs["-p"]
        self.bug_id = configs["-i"]
        self.experiment = configs["-e"]
        self.method = configs["-m"].split(",")
        self.dataloader = self._choose_dataloader_obj()
        self._update_cp()



    def _update_cp(self):
        if self.dataloader.feature_df.shape[1] >= 5600:
            self.configs["-cp"] = 0.5
        elif self.dataloader.feature_df.shape[1] >= 3400:
            self.configs["-cp"] = 0.5
        elif self.dataloader.feature_df.shape[1] >= 2600:
            self.configs["-cp"] = 0.5
        elif self.dataloader.feature_df.shape[1] >= 2010:
            self.configs["-cp"] = 0.5
        elif self.dataloader.feature_df.shape[1] >= 1540:
            self.configs["-cp"] = 0.5
        elif self.dataloader.feature_df.shape[1] >= 1080:
            self.configs["-cp"] = 0.5
        elif self.dataloader.feature_df.shape[1] >= 670:
            self.configs["-cp"] = 0.5
        elif self.dataloader.feature_df.shape[1] >= 320:
            self.configs["-cp"] = 0.5
        elif self.dataloader.feature_df.shape[1] >= 190:
            self.configs["-cp"] = 0.5
        else:
            self.configs["cp"] = 0.5

    def run(self):
        self._run_task()

    def _dynamic_choose(self, loader):
        # self.dataset_dir = os.path.join(self.project_dir, "data")
        data_obj = loader(self.project_dir, self.program, self.bug_id)
        data_obj.load()
        return data_obj

    def _choose_dataloader_obj(self):
        """
        返回一个经过处理的dataloader
        """
        if self.dataset == "d4j":
            return self._dynamic_choose(Defects4JDataLoader)
        if self.dataset == "manybugs" or self.dataset == "motivation":
            return self._dynamic_choose(ManyBugsDataLoader)
        if self.dataset == "SIR":
            return self._dynamic_choose(SIRDataLoader)
        if self.dataset == "d4j_sliced":
            return self._dynamic_choose(SlicedDataLoader)
        if self.dataset == "d4j_new":
            return self._dynamic_choose(NewD4jDataLoader)

    def _run_task(self):

        print("feature shape before [Data Augmentation]: ", self.dataloader.feature_df.shape)
        if self.experiment == "origin":
            self.data_obj = self.dataloader
        elif self.experiment == "resampling":
            self.data_obj = ResamplingData(self.dataloader)
            self.data_obj.process()
        elif self.experiment == "undersampling":
            self.data_obj = UndersamplingData(self.dataloader)
            self.data_obj.process()
        elif self.experiment == "smote":
            self.data_obj = SMOTEData(self.dataloader)
            self.data_obj.process()
        elif self.experiment == "cvae":
            self.data_obj = CVAESynthesisData(self.dataloader)
            self.data_obj.process()
        elif self.experiment == "fs":
            cp = float(self.configs["-cp"])
            ep = float(self.configs["-ep"])
            self.data_obj = PCAData(self.dataloader)
            self.data_obj.process(cp, ep)
        elif self.experiment == "slice_mixup":
            self.data_obj = Slice(self.dataloader)
            self.data_obj.process()
            self.data_obj = MixUp(self.data_obj)
            self.data_obj.process()
        elif self.experiment == "lda_smote":
            cp = float(self.configs["-cp"])
            ep = float(self.configs["-ep"])
            self.data_obj = LDAData(self.dataloader)
            self.data_obj.process(cp, ep)
            self.data_obj = SMOTEData(self.data_obj)
            self.data_obj.process()
        elif self.experiment == "fs_cgan":
            self.data_obj = Slice(self.dataloader)
            self.data_obj.process()
            self.data_obj = CGANSynthesisData(self.data_obj)
            self.data_obj.process()

        elif self.experiment == "fs_cvae":
            cp = float(self.configs["-cp"])
            ep = float(self.configs["-ep"])
            self.data_obj = PCAData(self.dataloader)
            self.data_obj.process(cp, ep)   # 进行 PCA 降维
            self.data_obj = CVAESynthesisData(self.data_obj)
            self.data_obj.process()     # 依然是self.data_df   feature_df    label_df
        elif self.experiment == "fs_ddpm":
            cp = float(self.configs["-cp"])
            ep = float(self.configs["-ep"])
            self.data_obj = DynamicSliceData(self.dataloader)

            if "MLP-FL" not in self.method and "CNN-FL" not in self.method and "RNN-FL" not in self.method:
                self.data_obj.process()   # 进行 PCA 降维
            else:
                self.data_obj.process(True)
            self.data_obj = ConDiffusionSynData(self.data_obj)
            self.data_obj.process()     # 依然是self.data_df   feature_df    label_df

            # print("synthesis shape: ", self.data_obj.data_df)

            # cp = float(self.configs["-cp"])
            # ep = float(self.configs["-ep"])
            # self.data_obj = PCAData(self.dataloader)
            # self.data_obj.process(cp, ep)   # 进行 PCA 降维

        # print(self.data_obj.feature_df)

        print("feature shape after [Data Augmentation]: ", self.data_obj.feature_df.shape)
        # save_rank_path = os.path.join(self.project_dir, "results")
        save_rank_path = r"/home/zhangxiaohong/yangjunzhe/temp_F/Code_FL/results"
        cc = CalculateSuspiciousness(self.data_obj, self.method, save_rank_path, self.experiment)
        cc.run()
