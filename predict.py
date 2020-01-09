import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from utils.dataset import PredictDataset

from torch.utils.data import DataLoader

import time

import threading

class Prefetcher:
    def __init__(self):
        self._running = True

    def terminate(self):
        self._running = False

    def prefetch_data(self, loader, batch_list, num):
        loader_iter = iter(loader)
        while self._running:
            if len(batch_list) < num:
                try: 
                    batch_list.append(next(loader_iter))
                except StopIteration:
                    return
            else:
                time.sleep(0.2)
    

def predict_batch(net,
                path,
                device,
                scale_factor=1,
                out_threshold=0.5, batch_size=1, prefetch=5):

    dataset = PredictDataset(path, scale_factor)
    predict_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    batches = []

    p = Prefetcher()
    prefetcher = threading.Thread(target=p.prefetch_data, args=(predict_loader, batches, prefetch))
    try:
        prefetcher.start()
        net.eval()
        #for batch in predict_loader:
        while prefetcher.is_alive():
            if not len(batches):
                continue
            batch = batches.pop(0)
            with torch.no_grad():
                imgs = batch['image']
                imgs_name = batch['filename']
                imgs_size = batch['size']
                imgs = imgs.to(device=device, dtype=torch.float32)

                ## for batch image
                #outputs = net(imgs)
                for i in range(batch_size):

                    ## for batch image
                    #output = outputs[i]

                    #print(imgs.shape)
                    ## for one image
                    img = imgs[i]
                    img = img.unsqueeze(0)
                    output = net(img)

                    img_name = imgs_name[i]
                    origin_img_size = imgs_size[i].item()
                    if net.n_classes > 1:
                        probs = F.softmax(output, dim=1)
                    else:
                        probs = torch.sigmoid(output)
                    probs = probs.squeeze(0)
                    tf = transforms.Compose(
                        [
                            transforms.ToPILImage(),
                            transforms.Resize(origin_img_size),
                            transforms.ToTensor()
                        ]
                    )
                    probs = tf(probs.cpu())
                    full_mask = probs.squeeze().cpu().numpy()

                    yield (full_mask > out_threshold, img_name)
        prefetcher.join()
    except KeyboardInterrupt as e:
        p.terminate()
    except RuntimeError as e:
        try:
            logging.error(str(e))
            p.terminate()
            sys.exit(0)
        except SystemExit:
            os._exit(0)

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
    return full_mask > out_threshold

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', type=str,
                        help='directory of input images', required=True)
    parser.add_argument('--output', '-o', type=str,
                        help='directory of ouput images')
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)
    parser.add_argument('--overwrite', action="store_true",
                        help='overwrite the output directory?')
    parser.add_argument('--batch_size', '-b', type=int, default=None,
                        help='run batch size')
    parser.add_argument('--prefetch', type=int, default=5,
                        help='prefetch data')
    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    in_path = args.input

    if not os.path.isdir(in_path):
        logging.error(in_path + ' is not a directory.')
        exit(1)

    out_path = args.output
    try:
        os.makedirs(out_path)
    except FileExistsError:
        if not args.overwrite:
            logging.error(out_path + ' is existed already.')
            exit(1)
        
    except PermissionError:
        logging.error('no write permission to ' + out_path)
        exit(1)

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    t_start = time.time()
    if args.batch_size:
        for f_idx, (mask, f_name) in enumerate(predict_batch(net=net,
                               path=in_path,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device, 
                               batch_size=args.batch_size,
                               prefetch=args.prefetch)):
            logging.info("{:.2f} [{}] Predicting image {} ...".format(time.time()-t_start, f_idx, f_name))
            result = mask_to_image(mask)
            result.save(out_path + '/' + f_name)
    else:
        for f_idx, f_name in enumerate(os.listdir(in_path)):
            f_path = os.path.join(in_path, f_name)
            if os.path.isdir(f_path):
                continue
            logging.info("{:.2f} [{}] Predicting image {} ...".format(time.time()-t_start, f_idx, f_name))
            img = Image.open(f_path)
            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            result = mask_to_image(mask)
            result.save(out_path + '/' + f_name)

