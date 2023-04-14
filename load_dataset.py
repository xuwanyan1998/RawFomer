import rawpy
from torch.utils.data import Dataset
import tqdm
import random
import imageio
import numpy as np
import torch

def image_read_SID(short_expo_files, long_expo_files):
    """
    load image data to CPU ram
    input: (short exposure images' path list, long exposure images' path list)
    output: datalist
    """
    short_list = []
    long_list = []

    for i in tqdm.tqdm(range(len(short_expo_files))):

        raw = rawpy.imread(short_expo_files[i])
        img_short = raw.raw_image_visible.copy()
        raw.close()
        # img_short = (np.maximum(img - 512, 0) / (16383 - 512))
        short_list.append(img_short)

        raw = rawpy.imread(long_expo_files[i])
        img_long = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16).copy()
        raw.close()
        # img_long = np.float32(img / 65535.0)
        long_list.append(img_long)
    return short_list, long_list

class load_data_SID(Dataset):
    """Loads the Data."""

    def __init__(self, short_expo_files, long_expo_files, patch_size=512,training=True):

        self.training = training
        self.patch_size = patch_size
        self.long_expo_files = long_expo_files
        if self.training:
            print('\n...... Train files loading\n')
            self.short_list, self.long_list = image_read_SID(short_expo_files, long_expo_files)
            print('\nTrain files loaded ......\n')
        else:
            print('\n...... Test files loading\n')
            self.short_list, self.long_list = image_read_SID(short_expo_files, long_expo_files)
            print('\nTest files loaded ......\n')

    def __len__(self):
        return len(self.short_list)

    def __getitem__(self, idx):

        img_short = self.short_list[idx]
        img_long = self.long_list[idx]

        H, W = img_short.shape

        # if training: crop image to 512*512
        # if testing: use whole image
        if self.training:
            i = random.randint(0, (H - self.patch_size - 2) // 2) * 2
            j = random.randint(0, (W - self.patch_size - 2) // 2) * 2

            img_short_crop = img_short[i:i + self.patch_size, j:j + self.patch_size]
            img_long_crop = img_long[i:i + self.patch_size, j:j + self.patch_size, :]

            if random.randint(0, 100) > 50:
                img_short_crop = np.fliplr(img_short_crop).copy()
                img_long_crop = np.fliplr(img_long_crop).copy()

            if random.randint(0, 100) < 20:
                img_short_crop = np.flipud(img_short_crop).copy()
                img_long_crop = np.flipud(img_long_crop).copy()

        else:
            img_short_crop = img_short
            img_long_crop = img_long

        if self.long_expo_files[idx][-7] == '3':
            ap = 300
        else:
            ap = 100

        img_short_crop = (np.maximum(img_short_crop.astype(np.float32) - 512, 0))/(16383 - 512)* ap
        img_long_crop = img_long_crop/65535

        img_short = torch.from_numpy(img_short_crop).float().unsqueeze(0)
        img_long = torch.from_numpy((np.transpose(img_long_crop, [2, 0, 1]))).float()

        return img_short, img_long

def image_read_MCR(train_c_path, train_rgb_path):
    """
    load image data to CPU ram, our dataset cost about 30Gb ram for training.
    if you don't have enough ram, just move this "image_read" operation to "load_data"
    it will read images from path in patch everytime.
    input: (color raw images' path list, mono raw images' path list, RGB GT images' path list)
    output: datalist
    """
    gt_list = []
    inp_list = []

    for i in tqdm.tqdm(range(len(train_c_path))):
        color_raw = imageio.imread(train_c_path[i])
        inp_list.append(color_raw)

        gt_rgb = imageio.imread(train_rgb_path[i])
        gt_list.append(gt_rgb)

    return inp_list, gt_list, train_c_path

class load_data_MCR(Dataset):
    """Loads the Data."""

    def __init__(self, train_c_path, train_rgb_path, patch_size=512, training=True):

        self.training = training
        self.patch_size = patch_size
        if self.training:
            print('\n...... Train files loading\n')
            self.inp_list, self.gt_list, self.train_c_path = image_read_MCR(train_c_path, train_rgb_path)
            print('\nTrain files loaded ......\n')
        else:
            print('\n...... Test files loading\n')
            self.inp_list, self.gt_list, self.train_c_path = image_read_MCR(train_c_path, train_rgb_path)
            print('\nTest files loaded ......\n')

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        gt_rgb_image = self.gt_list[idx]
        inp_raw_image = self.inp_list[idx]

        img_num = int(self.train_c_path[idx][-23:-20])
        img_expo = int(self.train_c_path[idx][-8:-4],16)
        H, W = inp_raw_image.shape

        if img_num < 500:
            gt_expo = 12287
        else:
            gt_expo = 1023
        amp = gt_expo / img_expo

        inp_raw_image = (inp_raw_image / 255 * amp).astype(np.float32)
        gt_rgb_image = (gt_rgb_image / 255).astype(np.float32)

        if self.training:
            """
            if training, random crop and flip are employed.
            if testing, original image data will be used.
            """
            i = random.randint(0, (H - self.patch_size - 2) // 2) * 2
            j = random.randint(0, (W - self.patch_size  - 2) // 2) * 2

            inp_raw = inp_raw_image[i:i + self.patch_size , j:j + self.patch_size ]
            gt_rgb = gt_rgb_image[i:i + self.patch_size , j:j + self.patch_size , :]

            if random.randint(0, 100) > 50:
                inp_raw = np.fliplr(inp_raw).copy()
                gt_rgb = np.fliplr(gt_rgb).copy()

            if random.randint(0, 100) < 20:
                inp_raw = np.flipud(inp_raw).copy()
                gt_rgb = np.flipud(gt_rgb).copy()
        else:
            inp_raw = inp_raw_image
            gt_rgb = gt_rgb_image

        gt = torch.from_numpy((np.transpose(gt_rgb, [2, 0, 1]))).float()
        inp = torch.from_numpy(inp_raw).float().unsqueeze(0)

        return inp, gt
