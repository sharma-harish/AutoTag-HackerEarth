import pretrainedmodels
import torch
import albumentations
from dataset_test import AutoTagDatasetTest
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import joblib
import torch.nn as nn
from torch.nn import functional as F
import models
import os
import ast
import pdb

TEST_BAT_SIZE = int(os.environ.get('TEST_BAT_SIZE'))
MODEL_MEAN = ast.literal_eval(os.environ.get('MODEL_MEAN'))
MODEL_STD = ast.literal_eval(os.environ.get('MODEL_STD'))
IMG_HT = int(os.environ.get('IMG_HT'))
IMG_WD = int(os.environ.get('IMG_WD'))
DEVICE='cuda'

VALID_FOLDS = ast.literal_eval(os.environ.get('VALID_FOLDS'))

test_dataset = AutoTagDatasetTest(
    img_ht = IMG_HT,
    img_wd = IMG_WD,
    mean = MODEL_MEAN,
    std = MODEL_STD
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size = TEST_BAT_SIZE,
    shuffle = True,
    num_workers = 4
)

if __name__ == "__main__":
    predictions = {}

    for b, d in tqdm(enumerate(test_loader), total = int(len(test_dataset) / test_loader.batch_size)):
            # pdb.set_trace()
            image = d['image']
            image_id = d['image_id']

            image = image.to(DEVICE, dtype = torch.float)

            for x in range(5, 6):
                model = models.ResNet34(pretrained=False)
                model.load_state_dict(torch.load(f'F:\\Workspace\\AutoTag-HE\\models\\iteration_{x}.bin'))
                model.to(DEVICE)
                with torch.no_grad():
                    model.eval()
                
                output = model(image)
                
                for i in range(len(output)):
                    if x == 5:
                        predictions[image_id[i]] = output[i]
                    else:
                        predictions[image_id[i]] += output[i]
                
            # submission = []
            # for p in predictions.keys():
            #     # pdb.set_trace()
            #     mapping = {0: 'Food', 1: 'Attire', 2: 'misc', 3: 'Decorationandsignage'}
            #     submission.append((f'{p}', mapping[int(torch.argmax(predictions[p]))]))

            sub = pd.DataFrame(predictions, columns = ['Image', 'Class'])
            sub.to_csv(f'F:\\Workspace\\AutoTag-HE\\out\\submission_merged.csv', index=False)
                # predictions = predictions/5
        #         # pdb.set_trace()
        #         output = torch.argmax(output, axis = 1)
                
        
        #         # pdb.set_trace()
        #         output = [mapping[int(o)] for o in output]
        #         for x in range(len(output)):
        #             predictions.append((f'{image_id[x]}', output[x]))
        # submission = pd.DataFrame(predictions, columns = ['Image', 'Class'])
        # print(submission.head())

        # submission.to_csv(f'F:\\Workspace\\AutoTag-HE\\out\\submission_{VALID_FOLDS[0]}.csv', index=False)
    