import cv2
import numpy as np
import copy
import torch
from common.utils.etc_utils import rle_decode
from torch.utils.data import Dataset


class CTDataset(Dataset):
    def __init__(self, db, is_train, transforms=None):
        self.db = db.data
        self.is_train = is_train
        self.transforms = transforms

    def load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img.astype('float32')  # original is uint16
        mx = np.max(img)
        if mx:
            img /= mx  # scale image to [0, 1]
        return img

    def load_mask(self, annos, height, width):
        shape = (height, width, 3)
        mask = np.zeros(shape, dtype=np.uint8)
        for cat in annos:
            mask[..., cat['category_id']] = rle_decode(cat['segmentation'], shape[:2])
        return mask

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        data = copy.deepcopy(self.db[index])
        img = self.load_img(data['image_path'])
        if self.is_train:
            mask = self.load_mask(data['anno'], data['height'], data['width'])
            if self.transforms:
                transformed_data = self.transforms(image=img, mask=mask)
                img = transformed_data['image']
                mask = transformed_data['mask']
            img = np.transpose(img, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))
            return torch.tensor(img), torch.tensor(mask)

        else:
            if self.transforms:
                transformed_data = self.transforms(image=img)
                img = transformed_data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)
