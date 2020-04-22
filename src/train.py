import ast
import models
import os
import pandas as pd
import pdb
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import AutoTagDatasetTrain

DEVICE = 'cuda'

IMG_HT = int(os.environ.get('IMG_HT'))
IMG_WD = int(os.environ.get('IMG_WD'))
EPOCHS = int(os.environ.get('EPOCHS'))

TRAIN_BAT_SIZE = int(os.environ.get('TRAIN_BAT_SIZE'))
TEST_BAT_SIZE = int(os.environ.get('TEST_BAT_SIZE'))

MODEL_MEAN = ast.literal_eval(os.environ.get('MODEL_MEAN'))
MODEL_STD = ast.literal_eval(os.environ.get('MODEL_STD'))

TRAIN_FOLDS = ast.literal_eval(os.environ.get('TRAIN_FOLDS'))
VALID_FOLDS = ast.literal_eval(os.environ.get('VALID_FOLDS'))

def train(dataset, data_loader, model, optimizer):
    model.train()
    
    for b, d in tqdm(enumerate(data_loader), total = int(len(dataset) / data_loader.batch_size)):
        # pdb.set_trace()
        image = d['image']
        Class = d['Class']

        image = image.to(DEVICE, dtype = torch.float)
        Class = Class.to(DEVICE, dtype = torch.long)

        optimizer.zero_grad()

        output = model(image)
        # pdb.set_trace()
        loss = nn.CrossEntropyLoss()(output, Class)
        loss.backward()
        optimizer.step()

def eval(dataset, data_loader, model):
    model.eval()
    final_loss = 0
    counter = 0

    for b, d in tqdm(enumerate(data_loader), total = int(len(dataset) / data_loader.batch_size)):
        counter += 1
        image = d['image']
        Class = d['Class']

        image = image.to(DEVICE, dtype = torch.float)
        Class = Class.to(DEVICE, dtype = torch.long)

        output = model(image)
        loss = nn.CrossEntropyLoss()(output, Class)
        final_loss += loss

    return final_loss/counter

def main():
    model = models.ResNet34(pretrained = True)
    # pdb.set_trace()
    # improve on older model with higher size images
    # model = models.ResNet34(pretrained=False)
    # model.load_state_dict(torch.load('F:\\Workspace\\AutoTag-HE\\models\\iteration.bin'))
    model = model.to(DEVICE)
    # pdb.set_trace()
    train_dataset = AutoTagDatasetTrain(
        folds = TRAIN_FOLDS,
        img_ht = IMG_HT,
        img_wd = IMG_WD,
        mean = MODEL_MEAN,
        std = MODEL_STD
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size = TRAIN_BAT_SIZE,
        shuffle = True,
        num_workers = 4
    )

    valid_dataset = AutoTagDatasetTrain(
        folds = VALID_FOLDS,
        img_ht = IMG_HT,
        img_wd = IMG_WD,
        mean = MODEL_MEAN,
        std = MODEL_STD
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size = TEST_BAT_SIZE,
        shuffle = True,
        num_workers = 4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',
                patience = 5, factor = 0.3, verbose = True)

    for epoch in range(EPOCHS):
        train(train_dataset, train_loader, model, optimizer)
        with torch.no_grad():
            val_score = eval(valid_dataset, valid_loader, model)
        scheduler.step(val_score)
        torch.save(model.state_dict(), f'F:\\Workspace\\AutoTag-HE\\models\\iteration_{VALID_FOLDS[0]}.bin')

if __name__ == "__main__":
    main()