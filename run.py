import os
import sys

from pipeline.Pipeline import Pipeline
from utils.args_util import parse_args

import torch
import random
import numpy as np
def main():
    # 获得当前文件的父目录地址
    # D:\university_study\科研\slice\code\python\ICSE2022FLCode\ICSE2022FLCode-master
    project_dir = os.path.dirname(__file__)
    print(project_dir)

    # # 设置随机种子
    # seed = 2024
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # random.seed(seed)
    # np.random.seed(seed)

    sys.argv = ["run.py", "-d", "d4j_new", "-p", "Lang", "-i", "3", "-m",  "dstar,ochiai,barinel,MLP-FL,CNN-FL,RNN-FL", "-e", "fs_ddpm", "-cp", "0.7", "-ep", "0.7"]
    # sys.argv = ["run.py", "-d", "SIR", "-p", "space", "-i", "1", "-m",  "dstar,ochiai,barinel,MLP-FL,CNN-FL,RNN-FL", "-e", "fs_ddpm", "-cp", "0.7", "-ep", "0.7"] 
# all_p = ["Mockito","Time","Lang","Math","Chart" ]
    all_p = ["Lang", "Math","Time", "Mockito","Chart"]
    # all_p = ["Mockito","Time","Lang","Math","Chart" ]
    # all_i ={"Lang": [6, 10, 17, 26, 32, 33, 38, 39, 42, 49, 60, 63, 64],
    #        "Mockito": [3,6,10,11,12,13,14,16,17,18,20,21,22,23,24,25,27,28,32,33,34,35,37]}
    adjust_i ={"Lang": [19, 30, 31, 40, 43, 44,47],
                "Math": [4, 22, 44,45,53,55,57,65,67,69,92,95,96,100,101,106]}
    chart_list = list(range(0, 4))
    for i in range(5, 26):
        chart_list.append(i)
    all_i ={"Lang": list(range(66))[1:],
            "Chart": list(range(27))[1:26],
            "Mockito": list(range(39))[1:],
            "Time": list(range(28))[1:],
            "Closure": list(range(134))[1:],
            "Math": list(range(107))[1:],


            "gzip": list(range(6))[4:],
            "python": list(range(5))[1:],
            "libtiff": list(range(8))[1:],
            "space":[ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
            }

    # methods = [ "MLP-FL,CNN-FL,RNN-FL", "dstar,ochiai,barinel",]
    methods = ["dstar,ochiai,barinel,MLP-FL,CNN-FL,RNN-FL"]
    for p in all_p:
        sys.argv[4] = p
        sys.argv[8] = methods[0]
        for i in all_i[p]:
            for e in ["fs_ddpm"]:   # adjust
            # for e in ["origin","resampling","undersampling"]:
                sys.argv[6] = str(i)
                sys.argv[10] = e
                print("test: " + sys.argv[4] + "-" + sys.argv[6] + "-" + sys.argv[10])
                configs = parse_args(sys.argv)   # 返回一个 参数字典
                pl = Pipeline(project_dir, configs)
                pl.run()
                print()


if __name__ == "__main__":
    main()
