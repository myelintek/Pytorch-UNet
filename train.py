import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
#from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

from torchsummary import summary

import time

dir_img = '/mlsteam/input/car_masking/train/'
dir_mask = '/mlsteam/input/car_masking/train_masks/'
dir_checkpoint = './checkpoints/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              val_epoch=1.0,
              gpu_list=None):

    if (isinstance(net, nn.DataParallel)):
        origin_net = net.module
    else:
        origin_net = net

    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # get first image's shape, pass into summary
    summary(origin_net, dataset.__getitem__(0)['image'].size(), batch_size=batch_size//len(gpu_list))

    net.to(device=device)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    
    # ****** can't use worker, or docker specify --ipc=host ******
    #https://github.com/ultralytics/yolov3/issues/283
    #train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    #val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')


    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    if origin_net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    t_start = time.time()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        #with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        if True:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == origin_net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if origin_net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                #pbar.set_postfix(**{'loss (batch)': loss.item()})
                print("{:.2f} Training: epoch {:6.4f}, loss {} ".format(time.time()-t_start, (global_step*batch_size/n_train), loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train * val_epoch // batch_size) == 0:
                    val_score = eval_net(net, val_loader, device, n_val)
                    if origin_net.n_classes > 1:
                        #logging.info('Validation cross entropy: {}'.format(val_score))
                        print("{:.2f} Validation: epoch {:6.4f}, cross_entropy {} ".format(time.time()-t_start, (global_step * batch_size/n_train), val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)

                    else:
                        #logging.info('Validation Dice Coeff: {}'.format(val_score))
                        print("{:.2f} Validation: epoch {:6.4f}, Dice_Coeff {} ".format(time.time()-t_start, (global_step * batch_size/n_train), val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    # spent to much resource
                    #writer.add_images('images', imgs, global_step)
                    #if origin_net.n_classes == 1:
                    #    writer.add_images('masks/true', true_masks, global_step)
                    #    writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(origin_net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


class CheckGPUs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not torch.cuda.is_available():
            raise argparse.ArgumentError(self, "No GPU available.")
        gpus = [int(i) for i in set(values.split(','))]
        for g in gpus:
            if not 0 < g < torch.cuda.device_count():
                raise argparse.ArgumentError(self, "GPU number invalid: "+g)
        setattr(namespace, self.dest, gpus)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    parser.add_argument('--validation-epoch', dest='val_epoch', type=float, default=1.0,
                        help='Period epochs of Validation')
    parser.add_argument('-g', '--gpu-list', dest='gpu_list', type=str, default=None, action=CheckGPUs,
                        help='Specify visibal GPUs')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1)
    if device.type == 'cuda':
        net.cuda()

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    if device.type == 'cuda':
        if torch.cuda.device_count() > 1:
            if args.gpu_list:
                device=torch.device(args.gpu_list[0])
                if len(args.gpu_list) > 1:
                    net = nn.DataParallel(net, device_ids=args.gpu_list)
            else:
                net = nn.DataParallel(net)
    
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  val_epoch=args.val_epoch,
                  gpu_list=args.gpu_list)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
