import ast
from dataset import AutoTagDatasetTrain
import matplotlib.pyplot as plt
import os
import pdb

IMG_HT = int(os.environ.get('IMG_HT'))
IMG_WD = int(os.environ.get('IMG_WD'))

MODEL_MEAN = ast.literal_eval(os.environ.get('MODEL_MEAN'))
MODEL_STD = ast.literal_eval(os.environ.get('MODEL_STD'))

dataset = AutoTagDatasetTrain(
        folds = list(range(0,9)),
        img_ht = IMG_HT,
        img_wd = IMG_WD,
        mean = MODEL_MEAN,
        std = MODEL_STD
)

fig = plt.figure()

for i in range(len(dataset)):
    data = dataset[i]
    sample = data['image']
    mapping = {0: 'Food', 1: 'Attire', 2: 'misc', 3: 'Decorationandsignage'}
    # pdb.set_trace()
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title(mapping[int(data['Class'])])
    ax.axis('off')
    plt.imshow(sample)

    if i == 3:
        plt.show()
        break
