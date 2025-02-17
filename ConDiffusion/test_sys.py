import torch
import torchvision.models as models
from ConDiffusion.ddpm_conditional import Diffusion
from ConDiffusion.modules import UNet_conditional
import os

model = UNet_conditional(num_classes=2, img_size=(1,64), device = "cpu").to("cpu")
model.load_state_dict(torch.load(os.path.join(r"D:\university_study\科研\slice\code\python\ICSE2022FLCode\ICSE2022FLCode-master\ConDiffusion\models",
                                              "DDPM_conditional", "ckpt.pt")))
diffusion = Diffusion(img_size=(1,64), device="cpu")
labels = torch.arange(2).long().to("cpu")
with torch.no_grad():
    sampled_images = diffusion.sample(model, n=len(labels), labels=labels)

print(sampled_images)