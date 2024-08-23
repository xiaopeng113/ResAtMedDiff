import os
import random

import torch.nn as nn
from tqdm import tqdm
import numpy as np

from dice_loss.dice_loss import  MAELoss
from loss import MixedLoss
from multi_train_utils.distributed_utils import is_main_process
from utils import *
import einops
import logging
import timm


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_condition_decoder(self, model, n, image, in_ch=1, is_train=True):
        logging.info(f"Sampling {n} new images....")
        if not is_train:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            model.eval()

        with torch.no_grad():

            x = torch.randn((n, in_ch, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                predicted_noise = model(x, t, image)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        if is_train:
            model.train()
        return x


def train(device, model, dataloader, optimizer, diffusion, epoch):
    mse = nn.MSELoss()
    loss_fn = MixedLoss(weight_nll=1.0, weight_dice=1.0, weight_jaccard=1.0)

    loss_mean = 0.0

    if is_main_process():
        dataloader = tqdm(dataloader, colour="green", ncols=80)
    for i, (images, lable) in enumerate(dataloader):
        images = images.to(device)

        lable = lable.to(device)

        t = diffusion.sample_timesteps(lable.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(lable, t)
        scaler = torch.cuda.amp.GradScaler()
        # print(x_t.shape,noise.shape)
        with torch.cuda.amp.autocast():
            predicted_noise = model(x_t, t, images)
            loss = loss_fn(predicted_noise, noise)
            # loss = mse(predicted_noise, noise)
            loss_mean = loss_mean + loss.item()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr = optimizer.param_groups[0]["lr"]
       
        if is_main_process():
            dataloader.desc = "epoch: {} loss_mean: {} lr: {}".format(epoch, round(loss_mean / (i + 1), 3), round(lr, 5))

    return loss_mean

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        # lrf=0.001,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)






