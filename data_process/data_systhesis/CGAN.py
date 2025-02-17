import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from data_process.ProcessedData import ProcessedData
from data_process.data_systhesis.CVAE_model import *

""" 
       Definition discriminator
    """


class discriminator(nn.Module):
    def __init__(self, INUNITS):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(INUNITS, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        return x

    """ 
    Definition generator
    """


class generator(nn.Module):
    def __init__(self, INUNITS):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, INUNITS),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x


class CGANSynthesisData(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = raw_data.rest_columns

    def process(self):
        if len(self.label_df) < 2:
            return

        equal_zero_index = (self.label_df != 1).values
        equal_one_index = ~equal_zero_index

        pass_feature = np.array(self.feature_df[equal_zero_index])
        fail_feature = np.array(self.feature_df[equal_one_index])

        diff_num = len(pass_feature) - len(fail_feature)

        if diff_num < 1:
            return

        # 超参数
        num_epoch = 1000

        learning_rate = 0.0003
        batch_size = 32  # 批次大小

        # 假设输入数据的维度 (INUNITS) 为 20

        min_batch = 40
        batch_size = min_batch if len(self.label_df) >= min_batch else len(self.label_df)
        torch_dataset = Data.TensorDataset(torch.tensor(self.feature_df.values, dtype=torch.float32),
                                           torch.tensor(self.label_df.values, dtype=torch.int64))
        loader = Data.DataLoader(dataset=torch_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 )
        INUNITS = len(self.feature_df.values[0])
        z_dimension = 100  # 随机噪声维度
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        # 定义生成器类
        class Generator(nn.Module):
            def __init__(self, output_units):
                super(Generator, self).__init__()
                self.gen = nn.Sequential(
                    nn.Linear(100, 256),
                    nn.ReLU(True),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Linear(256, output_units),
                    nn.Tanh()
                )

            def forward(self, x):
                x = self.gen(x)
                return x

        # 定义判别器类
        class Discriminator(nn.Module):
            def __init__(self, input_units):
                super(Discriminator, self).__init__()
                self.dis = nn.Sequential(
                    nn.Linear(input_units, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                x = self.dis(x)
                return x

        # 初始化生成器和判别器
        G = Generator(output_units=INUNITS)
        D = Discriminator(input_units=INUNITS)

        # 选择设备
        G.to(device)
        D.to(device)

        # 损失函数和优化器
        criterion = nn.BCELoss()
        d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)
        g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)

        # 生成随机的真实数据（用于模拟实际数据，可以根据实际需要替换）
        # 将 self.label_df 转换为布尔掩码
        mask = (self.label_df == 1).values.ravel()  # 将 DataFrame 转换为布尔数组并展平

        # 使用布尔掩码过滤 self.feature_df 并将结果转换为 Tensor
        real_data = torch.tensor(self.feature_df[mask].values, dtype=torch.float32).to(device)

        # 计算符合条件的样本数量
        testLen = mask.sum()
        # 训练 GAN
        for epoch in range(num_epoch):
            # === 训练判别器 ===
            # 生成真实标签和假标签
            real_labels = torch.ones(testLen, 1).to(device)
            fake_labels = torch.zeros(testLen, 1).to(device)

            # 计算判别器在真实数据上的损失
            real_outputs = D(real_data)
            d_loss_real = criterion(real_outputs, real_labels)

            # 生成假数据并计算判别器在假数据上的损失
            z = torch.randn(testLen, z_dimension).to(device)
            fake_data = G(z)
            fake_outputs = D(fake_data)
            d_loss_fake = criterion(fake_outputs, fake_labels)

            # 判别器总损失
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # === 训练生成器 ===
            # 生成新的假数据并计算生成器的损失
            z = torch.randn(testLen, z_dimension).to(device)
            fake_data = G(z)
            outputs = D(fake_data)
            g_loss = criterion(outputs, real_labels)  # 生成器的目标是让判别器认为假数据是真实的

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # 打印训练进度
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epoch}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, '
                      f'D real: {real_outputs.data.mean():.4f}, D fake: {fake_outputs.data.mean():.4f}')

        # 生成新的假数据
        with torch.no_grad():
            z = torch.randn(diff_num, z_dimension).to(device)
            generated_data = G(z)
            print("Generated Data:")
            print(generated_data.shape)

        features_np = np.array(self.feature_df)
        print(features_np.shape)

        compose_feature = np.concatenate((features_np, generated_data.cpu().numpy()), axis=0)

        label_np = np.array(self.label_df)
        gen_label = np.ones(diff_num).reshape((-1, 1))
        compose_label = np.vstack((label_np.reshape(-1, 1), gen_label))

        self.label_df = pd.DataFrame(compose_label, columns=['error'], dtype=float)
        self.feature_df = pd.DataFrame(compose_feature, columns=self.feature_df.columns, dtype=float)

        self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)


