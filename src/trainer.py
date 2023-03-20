from src.metrics import AverageMeter
from src.custom_dataloader import WheatDset, collate_fn
from src.model import DETRModel

import torch
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
import numpy as np
import albumentations as A
from src.data_augmentations import *
import cv2
import math

import os
import sys
sys.path.append('./detr/')

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
'''
code taken from github repo detr , 'code present in engine.py'
'''
IMG_SAVE_DIR = '/Users/anushkafernando/code/vit-example/loss_outputs/'

matcher = HungarianMatcher()

weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}

losses = ['labels', 'boxes', 'cardinality']

def train_fn(data_loader,model,criterion,optimizer,device,scheduler,weight_dict, BATCH_SIZE):
    model.train()
    criterion.train()

    summary_loss = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for step, (images, targets, image_ids) in enumerate(tk0):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        output = model(images)

        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        summary_loss.update(losses.item(),BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss

def eval_fn(data_loader, model,criterion, device, weight_dict, BATCH_SIZE):
    sys.path.append('./detr/')
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets, image_ids) in enumerate(tk0):

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)

            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            summary_loss.update(losses.item(),BATCH_SIZE)
            tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss

def show_validation_score(train_loss_list, valid_loss_list, EPOCHS, save=False, save_dir=IMG_SAVE_DIR, save_name='objectdetection_validation_score.png', FOLD_NUM=1):
    fig = plt.figure(figsize=(10,10))
    train_loss = train_loss_list
    valid_loss = valid_loss_list

    ax = fig.add_subplot(title= 'losses')
    ax.plot(range(EPOCHS), train_loss, c='orange', label='train')
    ax.plot(range(EPOCHS), valid_loss, c='blue', label='valid')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()

    plt.tight_layout()
    if save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir+save_name)
    else:
        plt.show()

def run(df_folds, fold, EPOCHS, marking, BATCH_SIZE, num_classes,
                 num_queries, null_class_coef, LR):

    df_train = df_folds[df_folds['fold'] != fold]
    df_valid = df_folds[df_folds['fold'] == fold]

    train_dataset = WheatDset(
    image_ids=df_train.index.values,
    dataframe=marking,
    transforms=get_train_transforms()
    )

    valid_dataset = WheatDset(
    image_ids=df_valid.index.values,
    dataframe=marking,
    transforms=get_valid_transforms()
    )

    train_data_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
    )

    device = torch.device('mps')
    model = DETRModel(num_classes=num_classes,num_queries=num_queries)
    model = model.to(device)
    criterion = SetCriterion(num_classes-1, matcher, weight_dict, eos_coef = null_class_coef, losses=losses)
    criterion = criterion.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


    train_losses = []
    valid_losses = []

    for epoch in range(EPOCHS):
        train_loss = train_fn(train_data_loader, model,criterion, optimizer,device,scheduler=None, weight_dict=weight_dict, BATCH_SIZE =BATCH_SIZE)
        valid_loss = eval_fn(valid_data_loader, model,criterion, device, weight_dict=weight_dict, BATCH_SIZE=BATCH_SIZE)

        print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch+1,train_loss.avg,valid_loss.avg))
        torch.save(model.state_dict(), f'detr_{fold}_{epoch+1}.pth')

        # For visualization
        train_losses.append(train_loss.avg)
        valid_losses.append(valid_loss.avg)

    show_validation_score(train_losses, valid_losses, EPOCHS=EPOCHS)
