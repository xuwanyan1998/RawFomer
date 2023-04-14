import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

import numpy as np
import os
import tqdm
import imageio
import glob
from model import RawFormer
from load_dataset import load_data_MCR, load_data_SID

if __name__ == '__main__':
    opt = {}
    opt['dataset'] = 'MCR'
    opt['use_gpu'] = True
    opt['gpu_id'] = '0'
    opt['model_size'] = 'S'  # 32/48/64 --> small/base/large

    save_weights_file = os.path.join('result', opt['dataset'], 'weights')
    save_images_file = os.path.join('result', opt['dataset'], 'images')
    save_csv_file = os.path.join('result', opt['dataset'], 'csv')
    tb_log_dir = os.path.join('result', opt['dataset'], 'logs')

    if opt['dataset'] == 'SID':
        test_input_paths = glob.glob(os.path.join('Sony/short/1*_00_0.1s.ARW'))
        test_gt_paths = []
        for x in test_input_paths:
            test_gt_paths += glob.glob(os.path.join('Sony/long/*' + x[-17:-12] + '*.ARW'))
        print('test data: %d pairs'%len(test_input_paths))
        test_data = load_data_SID(test_input_paths, test_gt_paths,training=False)

    elif opt['dataset'] == 'MCR':
        test_c_path = np.load('Mono_Colored_RAW_Paired_DATASET/random_path_list/test/test_c_path.npy')
        test_rgb_path = np.load('Mono_Colored_RAW_Paired_DATASET/random_path_list/test/test_rgb_path.npy')
        print('test data: %d pairs' % len(test_c_path))
        test_data = load_data_MCR(test_c_path, test_rgb_path, training=False)

    dataloader_test = DataLoader(test_data,batch_size=1, shuffle=False, pin_memory=True)

    if opt['use_gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    if opt['model_size'] == 'S':
        dim = 32
    elif opt['model_size'] == 'B':
        dim = 48
    else:
        dim = 64

    model = RawFormer(dim=dim)
    checkpoint = torch.load(save_weights_file + '/RawFormer_' + opt['model_size'] + '_' +opt['dataset'] + '.pth', map_location=device)

    # model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
        strict=True)
    epoch = checkpoint['epoch']
    print('load model from epoch:',epoch)

    model = model.to(device)

    print('Device on cuda: {}'.format(next(model.parameters()).is_cuda))
    model.eval()

    with torch.no_grad():
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_test in enumerate(tqdm.tqdm(dataloader_test)):
            rgb_gt = (data_test[1].numpy().squeeze().transpose((1, 2, 0))*255).astype(np.uint8)
            input_raw = data_test[0].to(device)

            pred_rgb = model(input_raw)
            pred_rgb = (torch.clamp(pred_rgb, 0, 1).cpu().numpy().squeeze().transpose((1,2,0))*255).astype(np.uint8)
            psnr = PSNR(pred_rgb, rgb_gt)
            ssim = SSIM(pred_rgb, rgb_gt, multichannel=True)
            print('image:{}\tPSNR:{:.4f}\tSSIM:{:.4f}\t'.format(ii,psnr,ssim))
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)
            imageio.imwrite(os.path.join(save_images_file, 'e{}_{}_gt.jpg'.format(epoch, ii)), rgb_gt)
            imageio.imwrite(os.path.join(save_images_file,'e{}_{}_psnr_{:.4f}_ssim_{:.4f}.jpg'.format(epoch, ii, psnr, ssim)), pred_rgb)

    psnr_average = sum(psnr_val_rgb) / len(dataloader_test)
    ssim_average = sum(ssim_val_rgb) / len(dataloader_test)

    print("average_PSNR: %f, average_SSIM: %f " % (psnr_average, ssim_average))
    np.savetxt(os.path.join(save_csv_file, 'test_metrics.csv'), [p for p in zip(psnr_val_rgb, ssim_val_rgb)], delimiter=',', fmt='%s')