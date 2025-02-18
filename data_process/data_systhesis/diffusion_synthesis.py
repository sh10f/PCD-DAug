import os
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import torch.utils.data as Data

from ConDiffusion.utils import *
from ConDiffusion.modules import UNet_conditional, EMA
from ConDiffusion.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

import logging
from tensorboardX import SummaryWriter
from data_process.ProcessedData import ProcessedData

import pandas as pd

# from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(256, 256), device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new convMatrix....")
        model.eval()
        with torch.no_grad():
            # reset the x's shape from [n, 3, self.img_size,self.img_size] to (n, 1, self.img_size[0], self.img_size[1])
            x = torch.randn((n, 1, self.img_size[0], self.img_size[1])).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                # cfg ---- linear
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = torch.sigmoid(x)

        return x

def get_batch_size(data_length):
    if data_length > 10000:
        return 258
    elif data_length > 1000:
        return 128
    elif data_length > 500:
        return 128
    elif data_length > 250:
        return 64
    elif data_length > 125:
        return 32
    elif data_length > 65:
        return 16
    else:
        return 8

class ConDiffusionSynData(ProcessedData):
    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.diff_num = None
        self.dataloader = None
        # self.rest_columns = raw_data.rest_columns

    def process(self):
        equal_zero_index = (self.label_df != 1).values
        equal_one_index = ~equal_zero_index

        pass_feature = np.array(self.feature_df[equal_zero_index])
        fail_feature = np.array(self.feature_df[equal_one_index])


        self.diff_num = len(pass_feature) - len(fail_feature)

        if self.diff_num < 1 or len(fail_feature) <= 0:
            return

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        dataset_length = len(self.feature_df)
        t_batch_size = get_batch_size(dataset_length)
        
        self.launch(epoch=100, batch_size=t_batch_size,
                    img_num=len(self.feature_df), img_size=(1, len(self.feature_df.iloc[0])),
                    device=device, lr=1e-3)

        # 调整 self.feature_df 的形状，使其每一行变为 (1, 1, 632)
        features = torch.tensor(self.feature_df.values, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        labels = torch.tensor(self.label_df.values, dtype=torch.int64).squeeze()
        # 打印数据集的形状
        print("Dataset shape:", features.shape, labels.shape)

        # 创建 TensorDataset
        torch_dataset = Data.TensorDataset(features, labels)

        loader = Data.DataLoader(dataset=torch_dataset,
                                 batch_size=self.args.batch_size,
                                 shuffle=True,
                                 )

        self.dataloader = loader
        # 获取一个批次的数据
        data_iter = iter(self.dataloader)
        first_batch = next(data_iter)
        t_features, t_labels = first_batch

        # 打印数据集的形状
        print("Batch Dataloader shape:", t_features.shape, t_labels.shape)


        self.train()

        # 假设 systhesis_multiple 是你用来生成新特征的函数
        # sys_feature = self.systhesis_multiple(self.diff_num)
        sys_feature = self.systhesis_multiple(self.diff_num)
        sys_feature = sys_feature.cpu().numpy()

        # 生成与 sys_feature 行数相同的 sys_label，全为 1
        sys_label = np.ones((sys_feature.shape[0], 1))

        # 将 sys_feature 和 feature_df 拼接
        updated_feature_df = np.concatenate((self.feature_df.values, sys_feature), axis=0)

        # 将 sys_label 和 label_df 拼接
        updated_label_df = np.concatenate((self.label_df.values, sys_label), axis=0)

        # 更新 feature_df 和 label_df
        self.feature_df = pd.DataFrame(updated_feature_df, columns=self.feature_df.columns)
        self.label_df = pd.DataFrame(updated_label_df, columns=self.label_df.columns)
        self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)





    def train(self):
        global loss
        # Keeping a record of the losses for later viewing
        losses = []

        setup_logging(self.args.run_name)
        device = self.args.device
        # dataloader = get_data_matrix(self.args)
        model = UNet_conditional(num_classes=self.args.num_classes, img_size=self.args.image_size, device = device).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr)
        mse = nn.MSELoss()
        diffusion = Diffusion(img_size=self.args.image_size, device=device)

        # logger = SummaryWriter(os.path.join(r"D:\university_study\科研\slice\code\python\ICSEFLCode\ICSE2022FLCode-master\ConDiffusion\runs",
        #                                     self.program+"-"+self.bug_id))

        logger = SummaryWriter(os.path.join(r"../../ConDiffusion",
                                            "test"))
        # logger = SummaryWriter(os.path.join("runs", self.args.run_name))
        l = len(self.dataloader)
        # ema = EMA(0.995)
        # ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        print("\nstart train\nmodel save to: ", os.path.join("models",self.args.run_name,f"ckpt.pt"))
        temp_loss = 1e5
        temp_model = None
        for epoch in range(self.args.epochs):
            total_loss = 0
            if (epoch + 1) % 100 == 0:
                logging.info(f"Starting epoch {epoch}:")
                pbar = tqdm(self.dataloader)
            else:
                pbar = self.dataloader

            for i, (images, labels) in enumerate(pbar):
                images = images.to(device)
                # add the labels
                labels = labels.to(device)

                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(images, t)

                # CFG
                # set "labels == None" in the 10 percent time in which we only use the timestamp info.
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = model(x_t, t, labels)
                loss = mse(noise, predicted_noise)

                # Store the loss for later
                # losses.append(loss.item())
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # ema.step_ema(ema_model, model)
                if (epoch + 1) % 100 == 0:
                    pbar.set_postfix(MSE=loss.item())
                logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / l
                logging.info(f"Epoch {epoch} average loss: {avg_loss:.6f}")
            if total_loss < temp_loss:
                temp_loss = total_loss
                temp_model =  copy.deepcopy(model.state_dict())

            if (epoch+1) % 100 == 0:
                # add the labels
                # labels = torch.arange(2).long().to(device)

                # Print our the average of the last 100 loss values to get an idea of progress:
                # avg_loss = sum(losses[-100:]) / 100
                # print(f'\nFinished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}\n')

                # sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
                #
                # # ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
                #
                # print(sampled_images)


                # save_images(sampled_images, os.path.join("results", self.args.run_name, f"{epoch}.jpg"))
                # save_images(ema_sampled_images, os.path.join("results", self.args.run_name, f"{epoch}_ema.jpg"))
                torch.save(temp_model, r"./models/DDPM_conditional/ckpt.pt")
                # torch.save(ema_model.state_dict(), os.path.join("models", self.args.run_name, f"ema_ckpt.pt"))
                # torch.save(optimizer.state_dict(), os.path.join("models", self.args.run_name, f"optim.pt"))

        # View the loss curve
        # plt.plot(losses)
        # plt.show()
        logger.close()


    def launch(self, epoch, batch_size, img_num, img_size, device, lr):
        import argparse
        parser_t = argparse.ArgumentParser()
        self.args, unknown = parser_t.parse_known_args()
        self.args.run_name = "DDPM_conditional"
        self.args.epochs = epoch
        self.args.batch_size = batch_size
        self.args.image_num  = img_num
        self.args.image_size = img_size
        self.args.num_classes = 2
        self.args.device = device
        self.args.lr = lr

        # self.args.epochs = 1
        # self.args.batch_size = 64
        # self.args.image_num  = 128
        # self.args.image_size = (1, 48)
        # self.args.num_classes = 2
        # self.args.device = "cpu"
        # self.args.lr = 3e-4



    def train_model(self):
        self.train()

    def systhesis(self):
        model = UNet_conditional(num_classes=self.args.num_classes, img_size=self.args.image_size, device=self.args.device ).to(self.args.device )
        model.load_state_dict(torch.load(os.path.join(
            r"D:\university_study\科研\slice\code\python\ICSE2022FLCode\ICSE2022FLCode-master\ConDiffusion\models",
            "DDPM_conditional", "ckpt.pt")))
        diffusion = Diffusion(img_size=self.args.image_size, device=self.args.device )
        labels = torch.arange(2).long().to(self.args.device )
        with torch.no_grad():
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)

        print(sampled_images)
        return sampled_images

    def systhesis_multiple(self, sampled_num):
        model = UNet_conditional(num_classes=self.args.num_classes, img_size=self.args.image_size, device=self.args.device ).to(self.args.device )
        print("load from:/home/fushihao/diffusion/Code_FL/data_process/data_systhesis/models/DDPM_conditional/ckpt.pt ") 
        model.load_state_dict(torch.load(r"./models/DDPM_conditional/ckpt.pt"))
        # model.load_state_dict(torch.load(os.path.join(
        #     r"D:\university_study\科研\slice\code\python\ICSE2022FLCode\ICSE2022FLCode-master\ConDiffusion\models",
        #     "DDPM_conditional", "ckpt.pt")))

         # ！！！加入dpm—solver
        model_kwargs = {}   # 无额外的参数
        x_T = torch.randn((2, 1, self.args.image_size[0], self.args.image_size[1])).to(self.args.device)  # 标准正态分布
        condition = torch.arange(2).long().to(self.args.device)   # 0,1
        unconditional_condition = None

        noise_steps=1000
        beta_start=1e-4
        beta_end=0.02
        betas = torch.linspace(beta_start, beta_end, noise_steps)

        guidance_scale = 4

        # 1. Define the noise schedule.
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

        # 2. Convert your discrete-time `model` to the continuous-time
        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type="noise",  # or "x_start" or "v" or "score"
            model_kwargs=model_kwargs,
            guidance_type="classifier-free",
            condition=condition,
            unconditional_condition=unconditional_condition,
            guidance_scale=guidance_scale,
        )

        # 3. Define dpm-solver and sample by multistep DPM-Solver.
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")


        systhesis_sampels = torch.randn(size=self.args.image_size).to(self.args.device)
        logging.info(f"Total sampling {sampled_num} new convMatrix....")
        print()
        for i in range(sampled_num):
            print(f"\rNow sampling {i}st new convMatrix....", end=' ')
            with torch.no_grad():
                x_T = torch.randn((2, 1, self.args.image_size[0], self.args.image_size[1])).to(self.args.device)  # 标准正态分布
                tensor1 = dpm_solver.sample(
                    x_T,
                    steps=25,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            extracted_tensor = tensor1[1, 0, 0, :].unsqueeze(0).to(self.args.device)
            extracted_tensor = torch.sigmoid(extracted_tensor)
            extracted_tensor = (extracted_tensor >= 0.45).float()
            systhesis_sampels = torch.cat((systhesis_sampels, extracted_tensor), dim=0)


        sys_target = systhesis_sampels[1:]
        print(sys_target.shape)

        return sys_target

        # ！！！ dpm-solver end here


        # diffusion = Diffusion(img_size=self.args.image_size, device=self.args.device)
        #
        # labels = torch.arange(2).long().to(self.args.device )
        #
        # systhesis_sampels = torch.randn(size=self.args.image_size)
        # for i in range(sampled_num):
        #     with torch.no_grad():
        #         tensor1 = diffusion.sample(model, n=len(labels), labels=labels)
        #
        #     extracted_tensor = tensor1[1, 0, 0, :].unsqueeze(0)
        #     systhesis_sampels = torch.cat((systhesis_sampels, extracted_tensor), dim=0)
        #
        #
        # sys_target = systhesis_sampels[1:]
        # print(sys_target.shape)
        #
        # return sys_target









if __name__ == '__main__':
    import pandas as pd
    path = r"./NBA_stat_toPython.csv"
    data = pd.read_csv(path)

    mix = ConDiffusionSynData(data)
    mix.data_df = data
    mix.feature_df = data.iloc[:, 3:-1]
    mix.label_df = data.iloc[:, -1]
    mix.process()
    mix.data_df.to_csv(r"./a.csv", index=True)
    print(data.head())
    train = ConDiffusionSynData()
    train.launch(epoch=10, batch_size=64, img_num=128, img_size=(1,48), device="cpu", lr=3e-4)
    train.train_model()
    # sam = train.systhesis_multiple(2)
    # print(sam.shape)



    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

