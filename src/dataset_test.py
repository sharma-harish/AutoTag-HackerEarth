import albumentations
import ast
import numpy as np
import os
import pandas as pd
import pdb
from skimage import io
from PIL import Image
import torch

TEST_CSV=os.environ.get('TEST_CSV')
TEST_PATH=os.environ.get('TEST_PATH')

class AutoTagDatasetTest:
    def __init__(self, img_ht, img_wd, mean, std):
        df = pd.read_csv(TEST_CSV)
        
        self.image_ids = df.Image.values

        self.aug = albumentations.Compose([
            albumentations.Resize(img_ht, img_wd, always_apply = True),
            albumentations.Normalize(mean, std, always_apply = True)
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # pdb.set_trace()
        image = io.imread(TEST_PATH + self.image_ids[idx])
        image_id = self.image_ids[idx]
        image = Image.fromarray(image).convert('RGB')
        # print(self.image_ids[idx])
        # print(image.shape)
        image = self.aug(image = np.array(image))['image']
        
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # pdb.set_trace()
        return {
            'image_id': image_id,
            'image': torch.tensor(image, dtype = torch.float)
        }