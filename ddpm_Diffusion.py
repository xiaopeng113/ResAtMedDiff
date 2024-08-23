import argparse
import os

from torch import nn

from get_evaluation import evaluation
import einops
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from get_evaluation_ import calculate_dice_score, calculate_iou, calculate_segmentation_metrics
# from modules.modules_liman import UNet_conditional
from modules.modules_base import UNet_conditional


# from train_test import train_one_epoch, t

from train_sample import Diffusion, train, create_lr_scheduler

from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from utils import keep_image_size_open, keep_image_size_open_rgb, save_images_double, save_images, \
    save_images_single_channel, save_images_three, keep_image_size_open_L


class Get_tager_sample(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'masks'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  
        segment_path = os.path.join(self.path, 'masks', segment_name)
        image_path = os.path.join(self.path, 'images', segment_name)
        segment_image = keep_image_size_open(segment_path, size=(256, 256))
        image = keep_image_size_open_rgb(image_path, size=(256, 256))
        segment_image = torch.Tensor(np.array(segment_image))
        segment_image = torch.unsqueeze(segment_image, 0)
        image = torch.Tensor(np.array(image))
        image = einops.rearrange(image, "w h c ->  c w h")

        return image, segment_image / 255



def main(args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    # weights_path = args.weights
    args.lr *= args.world_size  
    checkpoint_path = ""

    if rank == 0:  
        print(args)
        print("initing over!")
        tb_write = SummaryWriter()
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    print(os.cpu_count())
    torch.cuda.empty_cache()
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    print(torch.cuda.memory_allocated() / 1024 / 1024)
    train_1 = Get_tager_sample(r"")
    test_1 = Get_tager_sample("")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_1)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_1, shuffle=True)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)
    test_batch_sampler = torch.utils.data.BatchSampler(
        test_sampler, batch_size, drop_last=True)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_1,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=28,
                                               )

    test_loader = torch.utils.data.DataLoader(test_1,
                                              batch_sampler=test_batch_sampler,
                                              pin_memory=True,
                                              num_workers=28,
                                              )


    net = UNet_conditional(c_in=1, c_out=1, device=device, con_c_in=3).to(device)

    checkpoint_path = "initial_weights.pt"
    if rank == 0:
        torch.save(net.state_dict(), checkpoint_path)
    dist.barrier()
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)
    diffusion = Diffusion(img_size=256, device=device)
    lr = 1e-5

    net_optim = optim.Adam(net.parameters(), lr=lr)


    flage = 0
    best_miou = 0
    for i in range(801):
        epoch = 0 + i
        torch.cuda.empty_cache()
        train_sampler.set_epoch(epoch)

        loss_mean = train(device=device, model=net, optimizer=net_optim, diffusion=diffusion, dataloader=train_loader,
                          epoch=epoch)
        if(epoch % 10 == 0 or epoch >= 300):
            if epoch == 1600:
                torch.save(net.module.state_dict(), 'brats_model/1600_v2.pth')
            if rank == 0:
                tb_write.add_scalar("mes", 1, epoch)
                # net = torch.nn.parallel.DistributedDataParallel(net, device_ids= 0, find_unused_parameters=True)
                img, lable = next(iter(test_loader))
                lable = lable.to(device)

                sampled_images = diffusion.sample_condition_decoder(net, n=img.shape[0], image=img)
                sampled_images = sampled_images.to(device)
                dice_score, iou_score = calculate_segmentation_metrics(sampled_images, lable)

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--syncBN', type=bool, default=False)
    parser.add_argument('--is_load_net', type=bool, default=False)

    parser.add_argument('--save_weight', type=str, default='weight_att',
                        help='save weights path')

    parser.add_argument('--save_img', type=str, default='img_att',
                        help='save img path')

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)
