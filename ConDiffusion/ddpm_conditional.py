import os
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from tensorboardX import SummaryWriter
from data_process.ProcessedData import ProcessedData

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


class train_and_sys():
    def __init__(self):
        pass
    # def __init__(self, raw_data):
    #     super().__init__(raw_data)
    #     # self.rest_columns = raw_data.rest_columns

    def train(self):
        global loss
        # Keeping a record of the losses for later viewing
        losses = []

        setup_logging(self.args.run_name)
        device = self.args.device
        dataloader = get_data_matrix(self.args)

        # 获取一个批次的数据
        data_iter = iter(dataloader)
        first_batch = next(data_iter)
        features, labels = first_batch

        # 打印数据集的形状
        print("Dataset shape:", features.shape, labels.shape)

        model = UNet_conditional(num_classes=self.args.num_classes, img_size=self.args.image_size, device = device).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr)
        mse = nn.MSELoss()
        diffusion = Diffusion(img_size=self.args.image_size, device=device)

        logger = SummaryWriter(os.path.join(r"D:\university_study\科研\slice\code\python\ICSEFLCode\ICSE2022FLCode-master\ConDiffusion\runs",
                                            self.args.run_name))
        # logger = SummaryWriter(os.path.join("runs", self.args.run_name))
        l = len(dataloader)
        # ema = EMA(0.995)
        # ema_model = copy.deepcopy(model).eval().requires_grad_(False)



        for epoch in range(self.args.epochs):
            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloader)
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
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # ema.step_ema(ema_model, model)

                pbar.set_postfix(MSE=loss.item())
                # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

            if (epoch+1) % 10 == 0:
                # add the labels
                labels = torch.arange(2).long().to(device)

                # Print our the average of the last 100 loss values to get an idea of progress:
                avg_loss = sum(losses[-100:]) / 100
                print(f'\nFinished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}\n')

                sampled_images = diffusion.sample(model, n=len(labels), labels=labels)

                # ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)

                print(sampled_images)
                # save_images(sampled_images, os.path.join("results", self.args.run_name, f"{epoch}.jpg"))
                # save_images(ema_sampled_images, os.path.join("results", self.args.run_name, f"{epoch}_ema.jpg"))
                torch.save(model.state_dict(), os.path.join("models", self.args.run_name, f"ckpt.pt"))
                # torch.save(ema_model.state_dict(), os.path.join("models", self.args.run_name, f"ema_ckpt.pt"))
                torch.save(optimizer.state_dict(), os.path.join("models", self.args.run_name, f"optim.pt"))

        # View the loss curve
        plt.plot(losses)
        plt.show()


    def launch(self, epoch, batch_size, img_num, img_size, device, lr):
        import argparse
        parser = argparse.ArgumentParser()
        self.args = parser.parse_args()
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
        systhesis_sampels = torch.randn(size=self.args.image_size)
        for i in range(sampled_num):
            tensor1 = self.systhesis()
            extracted_tensor = tensor1[1, 0, 0, :].unsqueeze(0)
            systhesis_sampels = torch.cat((systhesis_sampels, extracted_tensor), dim=0)


        sys_target = systhesis_sampels[1:]
        print(sys_target)
        print(sys_target.shape)

        return sys_target








if __name__ == '__main__':
    train = train_and_sys()
    train.launch(epoch=10, batch_size=64, img_num=128, img_size=(1,16), device="cpu", lr=3e-4)
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

