import os.path
from skimage.draw import polygon2mask
import cv2
import numpy as np
import copy
import torch
from common.utils.etc_utils import rle_decode
from torch.utils.data import Dataset
import pydicom
import random
random.seed(0)


class CTDataset(Dataset):
    def __init__(self, db, transforms=None):
        _data = copy.deepcopy(db.data)
        random.shuffle(_data)
        # self.db = _data[data_range[0]:data_range[1]]
        self.db = _data
        print(f'The number of data: {len(self.db)}')

        self.cat_name = db.cat_name
        self.transforms = transforms
        self.rle = db.rle
        self.color_space = 'HSV'

    def load_img(self, path, colorspace, extension='jpg'):
        if extension == 'jpg':
            img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        elif extension == 'dcm':
            dcm = pydicom.dcmread(path)
            dcm_img = dcm.pixel_array
            dcm_img = dcm_img.astype(float)
            # dcm_img = dcm_img * dcm.RescaleSlope + dcm.RescaleIntercept
            # Rescaling grey scale between 0-255
            dcm_img_scaled = (np.maximum(dcm_img, 0) / dcm_img.max()) * 255
            # Convert to uint
            dcm_img_scaled = np.uint8(dcm_img_scaled)

            img_bgr = cv2.cvtColor(dcm_img_scaled, cv2.COLOR_GRAY2BGR)
        if colorspace == 'HSV':
            input_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        else:
            input_img = img_bgr

        img = input_img.astype('float32')  # original is uint16
        mx = np.max(img)
        if mx:
            img /= mx  # scale image to [0, 1]
        return img, img_bgr

    def load_mask(self, annos, height, width, rle=True):
        shape = (height, width, len(self.cat_name))
        mask = np.zeros(shape, dtype=np.uint8)
        if rle:
            for cat in annos:
                mask[..., cat['category_id']] = rle_decode(cat['segmentation'], shape[:2])
        else:
            for cat in annos:
                mask[..., cat['category_id']] = polygon2mask((height, width), cat['segmentation'])
        return mask

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        data = copy.deepcopy(self.db[index])
        extension = data['image_path'].split('.')[-1]
        img, img_bgr = self.load_img(data['image_path'], colorspace=self.color_space, extension=extension)

        mask = self.load_mask(data['anno'], data['height'], data['width'], self.rle)
        if self.transforms:
            transformed_data = self.transforms(image=img, mask=mask)
            transformed_ori_data = self.transforms(image=img_bgr, mask=mask)
            img = transformed_data['image']
            mask = transformed_data['mask']
            img_bgr = transformed_ori_data['image']

        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return torch.tensor(img), torch.tensor(mask), torch.tensor(img_bgr)

        # else:
        #     if self.transforms:
        #         transformed_data = self.transforms(image=img)
        #         img = transformed_data['image']
        #     img = np.transpose(img, (2, 0, 1))
        #     return torch.tensor(img)
