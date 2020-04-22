import albumentations
import ast
import numpy as np
import os
import pandas as pd
import pdb
from skimage import io
from PIL import Image
import torch

KFOLD_CSV=os.environ.get('KFOLD_CSV')
TRAIN_PATH=os.environ.get('TRAIN_PATH')

class AutoTagDatasetTrain:
    def __init__(self, folds, img_ht, img_wd, mean, std):
        df = pd.read_csv(KFOLD_CSV)
        
        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids = df.Image.values
        self.Class = df.Class.values

        if len(folds) == 1:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_ht, img_wd, always_apply = True),
                albumentations.Normalize(mean, std, always_apply = True)
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_ht, img_wd, always_apply = True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                                scale_limit=0.1,
                                                rotate_limit=5,
                                                p = 0.9),
                albumentations.Normalize(mean, std, always_apply = True)
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # pdb.set_trace()
        image = io.imread(TRAIN_PATH + self.image_ids[idx])
        image = Image.fromarray(image).convert('RGB')
        # print(self.image_ids[idx])
        # print(image.shape)
        image = self.aug(image = np.array(image))['image']
        
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # pdb.set_trace()
        return {
            'image': torch.tensor(image, dtype = torch.float),
            'Class':torch.tensor(self.Class[idx], dtype = torch.float)
        }
