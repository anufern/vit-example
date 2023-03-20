from src.metrics import AverageMeter
from src.custom_dataloader import WheatDset, collate_fn
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
import torch
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
import numpy as np
import albumentations as A
from src.data_augmentations import *
import cv2

import sys


'''
code taken from github repo detr , 'code present in engine.py'
'''

matcher = HungarianMatcher()

weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}

losses = ['labels', 'boxes', 'cardinality']

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

def convert_to_pascal_od(bbox):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    return [x_min, y_min, x_max, y_max]

def bounding_box_intersection_over_union(box_predicted, box_truth):
    # get (x, y) coordinates of intersection of bounding boxes
    top_x_intersect = max(box_predicted[0], box_truth[0])
    top_y_intersect = max(box_predicted[1], box_truth[1])
    bottom_x_intersect = min(box_predicted[2], box_truth[2])
    bottom_y_intersect = min(box_predicted[3], box_truth[3])

    # calculate area of the intersection bb (bounding box)
    intersection_area = max(0, bottom_x_intersect - top_x_intersect + 1) * max(
        0, bottom_y_intersect - top_y_intersect + 1
    )

    # calculate area of the prediction bb and ground-truth bb
    box_predicted_area = (box_predicted[2] - box_predicted[0] + 1) * (
        box_predicted[3] - box_predicted[1] + 1
    )
    box_truth_area = (box_truth[2] - box_truth[0] + 1) * (
        box_truth[3] - box_truth[1] + 1
    )

    # calculate intersection over union by taking intersection
    # area and dividing it by the sum of predicted bb and ground truth
    # bb areas subtracted by  the interesection area

    # return ioU
    return intersection_area / float(
        box_predicted_area + box_truth_area - intersection_area
    )

def model_eval(df_valid,model,device, marking, BATCH_SIZE):
    '''
    Code taken from Peter's Kernel
    https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train
    '''
    valid_dataset = WheatDset(image_ids=df_valid.index.values,
                                 dataframe=marking,
                                 transforms=get_valid_transforms()
                                )

    valid_data_loader = DataLoader(
                                    valid_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                   num_workers=4,
                                   collate_fn=collate_fn)

    images, targets, image_ids = next(iter(valid_data_loader))

    i, mean_iou = 0, 0

    # Begin loop here

    for idx, image in enumerate(images):

        _,h,w = images[idx].shape # for de normalizing images

        images = list(img.to(device) for img in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        boxes = targets[idx]['boxes'].cpu().numpy()
        boxes = [np.array(box).astype(np.int32) for box in A.core.bbox_utils.denormalize_bboxes(boxes,h,w)]
        sample = images[idx].permute(1,2,0).cpu().numpy()

        model.eval()
        model.to(device)
        cpu_device = torch.device("cpu")

        with torch.no_grad():
            outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}]


        oboxes = outputs[0]['pred_boxes'][idx].detach().cpu().numpy()
        oboxes = [np.array(box).astype(np.int32) for box in A.core.bbox_utils.denormalize_bboxes(oboxes,h,w)]
        prob   = outputs[0]['pred_logits'][idx].softmax(1).detach().cpu().numpy()[:,0]



        #iou


        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        for gt_box, pred_box, p in zip(boxes, oboxes, prob):

        #for box in boxes:
            cv2.rectangle(sample,
                    (gt_box[0], gt_box[1]),
                    (gt_box[2]+gt_box[0], gt_box[3]+gt_box[1]),
                    (220, 0, 0), 1)





        #for box,p in zip(oboxes,prob):

            if p >0.5:
                color = (0,0,220) #if p>0.5 else (0,0,0)
                cv2.rectangle(sample,
                    (pred_box[0], pred_box[1]),
                    (pred_box[2]+pred_box[0], pred_box[3]+pred_box[1]),
                    color, 1)

                box_truth = convert_to_pascal_od(gt_box)
                box_predicted = convert_to_pascal_od(pred_box)
                iou = bounding_box_intersection_over_union(box_predicted, box_truth)
                mean_iou += bounding_box_intersection_over_union(box_predicted, box_truth)


        ax.set_axis_off()



        ax.imshow(sample, rasterized=True)


        fig.savefig('outputs/test_image_{}.png'.format(idx))
        # a colormap and a normalization instance
        #cmap = plt.cm.jet
        #norm = plt.Normalize(vmin=sample.min(), vmax=sample.max())

        # map the normalized data to colors
        # image is now RGBA (512x512x4)
        #norm_sample = cmap(norm(sample))

        # save the image

        #plt.imsave('~outputs/test_image_{}.png'.format(idx), norm_sample)

    print("mean_iou: " + str(mean_iou / len(images)))
