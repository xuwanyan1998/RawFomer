import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as PSNR
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import tqdm
import glob
import time
import datetime
from model import RawFormer
from load_dataset import load_data_MCR, load_data_SID


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

if __name__ == '__main__':
    opt = {}
    opt['gpu_id'] = '0'
    opt={'base_lr':1e-4}        # base learning rate
    opt['batch_size'] = 16      # batch size
    opt['dataset'] = 'SID'      # SID/MCR dataset
    opt['patch_size'] = 512     # cropped image patch size when training
    opt['model_size'] = 'S'     # model size, small/base/large --> 32/48/64
    opt['epochs'] = 3000        # total training epochs

    os.environ["CUDA_VISIBLE_DEVICES"]=opt['gpu_id']
    print('GPU id:', os.environ["CUDA_VISIBLE_DEVICES"])

    # These are folders
    save_weights_file = os.path.join('result', opt['dataset'], 'weights')   # save trained models
    save_images_file = os.path.join('result', opt['dataset'], 'images')     # save tested images
    save_csv_file = os.path.join('result', opt['dataset'], 'csv')           # save tested images' psnr/ssim
    tb_log_dir = os.path.join('result', opt['dataset'], 'logs')             # save trained logs

    if not os.path.exists(save_weights_file):
        os.makedirs(save_weights_file)
    if not os.path.exists(save_images_file):
        os.makedirs(save_images_file)
    if not os.path.exists(save_csv_file):
        os.makedirs(save_csv_file)
    if not os.path.exists(tb_log_dir):
        os.makedirs(tb_log_dir)

    use_pretrain = False
    pretrain_weights = os.path.join(save_weights_file, 'model_2000.pth')

    if opt['dataset'] == 'SID':
        train_input_paths = glob.glob(os.path.join('Sony/short/0*_00_0.1s.ARW')) + glob.glob(os.path.join('Sony/short/2*_00_0.1s.ARW'))
        train_gt_paths = []
        for x in train_input_paths:
            train_gt_paths += glob.glob(os.path.join('Sony/long/*' + x[-17:-12] + '*.ARW'))

        test_input_paths = glob.glob(os.path.join('Sony/short/1*_00_0.1s.ARW'))
        test_gt_paths = []
        for x in test_input_paths:
            test_gt_paths += glob.glob(os.path.join('Sony/long/*' + x[-17:-12] + '*.ARW'))

        # load data
        train_data = load_data_SID(train_input_paths, train_gt_paths, patch_size=opt['patch_size'], training=True)
        test_data = load_data_SID(test_input_paths, test_gt_paths, patch_size=opt['patch_size'], training=True)

    elif opt['dataset'] == 'MCR':
        train_c_path = np.load('Mono_Colored_RAW_Paired_DATASET/random_path_list/train/train_c_path.npy')
        train_rgb_path = np.load('Mono_Colored_RAW_Paired_DATASET/random_path_list/train/train_rgb_path.npy')
        test_c_path = np.load('Mono_Colored_RAW_Paired_DATASET/random_path_list/test/test_c_path.npy')
        test_rgb_path = np.load('Mono_Colored_RAW_Paired_DATASET/random_path_list/test/test_rgb_path.npy')
        # load data
        train_data = load_data_MCR(train_c_path[:32], train_rgb_path[:32], patch_size=opt['patch_size'], training=True)
        test_data = load_data_MCR(test_c_path[:32], test_rgb_path[:32], patch_size=opt['patch_size'], training=True)

    dataloader_train = DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True, num_workers=16, pin_memory=True)
    dataloader_val = DataLoader(test_data, batch_size=opt['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
    print('train data: %d batch'%len(dataloader_train))
    print('test data: %d batch'%len(dataloader_val))

    device = torch.device("cuda")

    if opt['model_size'] == 'S':
        dim = 32
    elif opt['model_size'] == 'B':
        dim = 48
    else:
        dim = 64

    model = RawFormer(dim=dim)

    print('\nTrainable parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('\nTotal parameters : {}\n'.format(sum(p.numel() for p in model.parameters())))
    model = model.to(device)
    print('Device on cuda: {}'.format(next(model.parameters()).is_cuda))

    start_epoch = 0
    end_epoch = opt['epochs']
    best_psnr = 0
    best_epoch = 0

    ######### Loss ###########
    loss_criterion = torch.nn.L1Loss()

    ######### Scheduler ###########
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['base_lr'])
    if use_pretrain:
        checkpoint = torch.load(pretrain_weights)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        start_epoch = checkpoint['epoch'] + 1

    print("Using warmup and cosine strategy!")
    warmup_epochs = 20
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, end_epoch-warmup_epochs, eta_min=1e-5)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

    torch.cuda.empty_cache()
    loss_scaler = torch.cuda.amp.GradScaler()    # 计算loss时用到的梯度scaler

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'valid_PSNR': 0,
        # 'valid_SSIM': 0,
        'best_PSNR': 0,
        'best_epoch': 0,
        'epoch_time': 0,
        'epoch_loss': 0,
        'epoch_LR': 0,
    }

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start_time = time.time()
        epoch_loss = 0

        for i, img in enumerate(tqdm.tqdm(dataloader_train)):
            optimizer.zero_grad()
            input_raw = img[0].to(device)
            gt_rgb = img[1].to(device)

            with torch.cuda.amp.autocast():
                pred_rgb = model(input_raw)
                pred_rgb = torch.clamp(pred_rgb, 0, 1)
                loss = loss_criterion(pred_rgb, gt_rgb)
            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
            epoch_loss += loss.item()

        scheduler.step()

        #### Evaluation ####
        with torch.no_grad():
            model.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate(tqdm.tqdm(dataloader_val)):
                input_raw = data_val[0].to(device)
                gt_rgb = data_val[1].to(device)
                with torch.cuda.amp.autocast():
                    pred_rgb = model(input_raw)
                pred_rgb = torch.clamp(pred_rgb, 0, 1)
                psnr_val_rgb.append(PSNR((data_val[1].numpy().transpose(0, 2, 3, 1)*255).astype(np.uint8),
                                         (pred_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1)*255).astype(np.uint8)))

            psnr_val_rgb = sum(psnr_val_rgb) / len(dataloader_val)

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(save_weights_file, "model_best.pth"))

            print("------------------------------------------------------------------")
            print("[PSNR SID: %.4f] ----  [best_Ep_SID: %d, Best_PSNR_SID: %.4f] " % (psnr_val_rgb, best_epoch, best_psnr))
            model.train()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        if writer_dict:
            writer = writer_dict['writer']
            writer.add_scalar('valid_PSNR', psnr_val_rgb, epoch)
            writer.add_scalar('best_PSNR', best_psnr, epoch)
            writer.add_scalar('best_epoch', best_epoch, epoch)
            writer.add_scalar('epoch_time', time.time() - epoch_start_time, epoch)
            writer.add_scalar('epoch_loss', epoch_loss, epoch)
            writer.add_scalar('epoch_LR', scheduler.get_lr()[0], epoch)

        if epoch == end_epoch:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(save_weights_file, "model_{}.pth".format(epoch)))

    print("Now time is : ", datetime.datetime.now().isoformat())
    print('Model saved in: ', save_weights_file)