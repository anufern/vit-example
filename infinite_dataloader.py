
import os
import numpy as np
import pandas as pd
import random
from src.custom_dataloader import *
from src.data_augmentations import *

from src.trainer import run
from src.evaluator import model_eval


#Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset

#sklearn
from sklearn.model_selection import StratifiedKFold

#CV
import cv2

#Box normalisation
import albumentations as A

#Glob
from glob import glob

DIR_TRAIN = "/Users/anushkafernando/code/vit-example/input/global-wheat-detection/train"


n_folds = 5
seed = 42

from torch.utils.data import DataLoader, Dataset, Sampler
import random
from itertools import cycle

class WheatDset(Dataset):
    def __init__(self,image_ids,dataframe,transforms=None):
        self.image_ids = image_ids
        self.df = dataframe
        self.transforms = transforms


    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self,index):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{DIR_TRAIN}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # DETR takes in data in coco format
        boxes = records[['x', 'y', 'w', 'h']].values

        #Area of bb
        area = boxes[:,2]*boxes[:,3]
        area = torch.as_tensor(area, dtype=torch.float32)

        # AS pointed out by PRVI It works better if the main class is labelled as zero
        labels =  np.zeros(len(boxes), dtype=np.int32)


        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']


        #Normalizing BBOXES

        _,h,w = image.shape
        boxes = A.core.bbox_utils.normalize_bboxes(sample['bboxes'],rows=h,cols=w)
        target = {}
        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels,dtype=torch.long)
        target['image_id'] = torch.tensor([index])
        target['area'] = area

        return image, target, image_id

class customSampler(Sampler) :
    def __init__(self, dataset, shuffle):
        assert len(dataset) > 0
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        order = list(range((len(self.dataset))))
        idx = 0
        while True:
            yield order[idx]
            idx += 1
            if idx == len(order):
                if self.shuffle:
                    random.shuffle(order)
                idx = 0

if __name__ == "__main__":
    marking = pd.read_csv('/Users/anushkafernando/code/vit-example/input/global-wheat-detection/train.csv')
    bboxs, marking = bboxs_marking(marking)
    df_folds = create_folds(n_folds, seed, marking)

    dset = WheatDset(image_ids=df_folds.index.values,
    dataframe=marking,
    transforms=get_train_transforms())
    sampler = customSampler(dset, shuffle=True)
    loader = iter(DataLoader(dataset=dset, sampler=sampler, batch_size=15, num_workers=2, collate_fn=collate_fn))

    # For demonstation
    #for x in range(10):
    #    i = next(loader)
    #    print(i)

    #For infinite loop

    for i, data in cycle(enumerate(loader)):
        print(i, data)
