import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        return data, target


def get_data_matrix(args):
    # 生成随机的 1000x50 的 覆盖矩阵
    coverage_matrix = np.random.randint(2, size=(args.image_num, args.image_size[1]))

    print("coverage_matrix.shape: ", coverage_matrix.shape)

    # 生成随机的 1000x1 的 error向量
    error_vector = np.random.randint(2, size=(args.image_num,))

    print("error_vector.shape: ", error_vector.shape)

    # 将 numpy 数组转换为 PyTorch 张量
    coverage_matrix_tensor = torch.tensor(coverage_matrix, dtype=torch.float32)
    error_vector_tensor = torch.tensor(error_vector, dtype=torch.int32)

    # 将 coverage_matrix_tensor 转换为形状为 Batch*1*1*width 的张量
    coverage_matrix_tensor = coverage_matrix_tensor.unsqueeze(1).unsqueeze(2)

    # 创建自定义数据集实例
    dataset = CustomDataset(coverage_matrix_tensor, error_vector_tensor)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 检查数据加载器
    for batch in dataloader:
        data, target = batch
        print("Data shape:", data.shape)
        print("Labels: ", target)
        print(target[1].dtype)
        print("Label shape:", target.shape)
        break  # 只检查第一个批次

    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 4
    get_data_matrix(args)
