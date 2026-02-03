import torchvision.transforms
import glob
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, Resize,Grayscale
import numbers
import random
from torchvision.transforms import functional as F
import os
from PIL import Image
import cv2


class FusionRandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self,img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)

        i, j, h, w = self.get_params(img, self.size)

        return (i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def train_hr_transform(crop_size):
    return Compose([
        FusionRandomCrop(crop_size),
    ])

def train_vis_ir_transform():
    return Compose([
        Grayscale(num_output_channels=3),
        ToTensor(),
    ])

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class FusionDataset(Dataset):
    def __init__(self,
                 split,
                 crop_size=128,  # resolution in training
                 min_max=(-1, 1),
                 MRI_path='/media/sata1/hedan/MSRS-main/test\ir',
                 vi_path='/media/sata1/hedan/MSRS-main/test/vi',
                 is_crop=True):
        super(FusionDataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.is_crop = is_crop
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor=1)
        self.hr_transform = train_hr_transform(crop_size)
        self.vis_ir_transform = train_vis_ir_transform()  # transform from rgb to grayscale
        self.min_max = min_max
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.1)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.1)

        if split == 'train':
            data_dir_vis = vi_path
            data_dir_ir = MRI_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'val':
            data_dir_vis = vi_path
            data_dir_ir = MRI_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split == 'train':
            visible_image = Image.open(self.filepath_vis[index]).convert ('L')
            infrared_image = Image.open(self.filepath_ir[index]).convert ('L')
            if self.is_crop:
                crop_size = self.hr_transform(visible_image)
                visible_image, infrared_image = F.crop(visible_image, crop_size[0], crop_size[1], crop_size[2],
                                                   crop_size[3]), \
                                                F.crop(infrared_image, crop_size[0], crop_size[1], crop_size[2], crop_size[3])
            # Random horizontal flipping
            if random.random() > 0.5:
                visible_image = self.hflip(visible_image)
                infrared_image = self.hflip(infrared_image)
            # Random vertical flipping
            if random.random() > 0.5:
                visible_image = self.vflip(visible_image)
                infrared_image = self.vflip(infrared_image)

            visible_image = ToTensor()(visible_image)*(self.min_max[1] - self.min_max[0]) + self.min_max[0]
            infrared_image = ToTensor()(infrared_image)*(self.min_max[1] - self.min_max[0]) + self.min_max[0]

            cat_img = torch.cat([visible_image, infrared_image[0:1, :, :]], axis=0)

            return {'img': cat_img, 'Other': visible_image, 'MRI': infrared_image[0:1, :, :]},  self.filenames_vis[index]

        elif self.split == 'val':
            visible_image = Image.open(self.filepath_vis[index]).convert ('L')
            # visible_image = cv2.imread (self.filepath_vis[index])
            # visible_image = cv2.cvtColor (visible_image, cv2.COLOR_BGR2RGB)
            infrared_image = Image.open(self.filepath_ir[index]).convert ('L')
            #
            visible_image = ToTensor()(visible_image)
            visible_image = visible_image*(self.min_max[1] - self.min_max[0]) + self.min_max[0]
            infrared_image = ToTensor()(infrared_image)
            infrared_image = infrared_image * (self.min_max[1] - self.min_max[0]) + self.min_max[0]

            cat_img = torch.cat([visible_image[:, :, :], infrared_image[0:1, :, :]], axis=0)

            return {'img': cat_img, 'Other': visible_image, 'MRI': infrared_image[0:1, :, :]}, self.filenames_vis[index]

    def __len__(self):
        return self.length
