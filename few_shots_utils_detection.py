import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import sys
import glob
from pycocotools.coco import COCO
import numpy as np
import torch
from random import randint
import random
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Subset, DataLoader
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR,_LRScheduler,CosineAnnealingWarmRestarts
import subprocess
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2

def generalized_box_iou_loss(PredBoxes, boxes2, eps=1e-7):
    """
    Generalized Intersection over Union Loss (GIoU)

    Gradient-friendly IoU loss with an additional penalty that is non-zero when the
    boxes do not overlap and scales with the size of their smallest enclosing box.

    Args:
        PredBoxes (Tensor): Predictions of bounding boxes, sized [batch_size, num_boxes, 4].
                         The boxes are expected to be in (x1, y1, x2, y2) format.
        boxes2 (Tensor): Ground truth bounding boxes, sized [batch_size, num_boxes, 4].
                         The boxes are expected to be in (x1, y1, x2, y2) format.
        eps (float): small number to prevent division by zero.

    Returns:
        Tensor: GIoU loss value for each box, sized [batch_size, num_boxes].
    """
    # Get the coordinates of bounding boxes
    x1_true, y1_true, x2_true, y2_true = boxes2.unbind(-1)
    x1_pred, y1_pred, x2_pred, y2_pred = PredBoxes.unbind(-1)

    # Calculate the intersection rectangle
    x1_intersection = torch.max(x1_true, x1_pred)
    y1_intersection = torch.max(y1_true, y1_pred)
    x2_intersection = torch.min(x2_true, x2_pred)
    y2_intersection = torch.min(y2_true, y2_pred)

    # Calculate the area of intersection rectangle
    intersection = (x2_intersection - x1_intersection) * \
                   (y2_intersection - y1_intersection)
    intersection = intersection.clamp(min=0,max=1)+eps

    # Calculate the area of bounding boxes
    area_true = (x2_true - x1_true) * (y2_true - y1_true)
    area_pred = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    area_pred = area_pred.clamp(min=0,max=1)

    # Calculate the union area
    union = area_true + area_pred - intersection
    union = union.clamp(min=0, max=1) + eps

    # Calculate the IoU
    iou = (intersection / union).clamp(min=0, max=1)

    # Calculate the smallest enclosing box
    x1_enclosing = torch.min(x1_true, x1_pred)
    y1_enclosing = torch.min(y1_true, y1_pred)
    x2_enclosing = torch.max(x2_true, x2_pred)
    y2_enclosing = torch.max(y2_true, y2_pred)

    # Calculate the area of the smallest enclosing box
    area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + eps

    if (area_enclosing<0).any() or (intersection<0).any() or (area_enclosing<0).any() or (area_pred<0).any():
        print("error ares")

    # Calculate the GIoU loss
    giou_loss = 1 - iou + (area_enclosing - union).clamp(min=0,max=1) / area_enclosing

    if (giou_loss<0).any():
        print("giou_loss error")

    return giou_loss.mean()


def compute_iou(PredBoxes, boxes2):
    """
    Compute the Intersection over Union (IoU) of two sets of boxes.

    Args:
        PredBoxes (torch.Tensor): Coordinates of bounding boxes of shape (B, 4), where B is the batch size and the coordinates are in the format [x1, y1, x2, y2].
        boxes2 (torch.Tensor): Coordinates of bounding boxes of shape (B, 4), where B is the batch size and the coordinates are in the format [x1, y1, x2, y2].

    Returns:
        torch.Tensor: IoU values of shape (B, B).
    """
    # Get the coordinates of the intersection rectangles
    x1 = torch.max(PredBoxes[:, 0], boxes2[:, 0])
    y1 = torch.max(PredBoxes[:, 1], boxes2[:, 1])
    x2 = torch.min(PredBoxes[:, 2], boxes2[:, 2])
    y2 = torch.min(PredBoxes[:, 3], boxes2[:, 3])

    # Compute the area of intersection rectangle
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Compute the area of both bounding boxes
    box1_area = (PredBoxes[:, 2] - PredBoxes[:, 0]).clamp(min=0) * (PredBoxes[:, 3] - PredBoxes[:, 1]).clamp(min=0)
    box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute the Intersection over Union by taking the intersection
    # area and dividing it by the sum of prediction and ground-truth
    # areas - the intersection area
    iou = intersection / (box1_area + box2_area - intersection + 1e-6)
    return iou







class ObjectDetectionAugmentation:
    def __init__(self, p,min_visibility,Mode):

        self.Mode=Mode

        self.zero_target_transform = A.Compose([
            A.Resize(224 * 2, 224 * 2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # EfficientNet and ResNet50
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=min_visibility))

        self.zero_query_Transform = A.Compose([
            A.Resize(128, 128),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # EfficientNet Resnet50
            A.pytorch.ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=min_visibility))


        self.pretrain_target_transform = A.Compose([
            A.RandomResizedCrop(height=224 * 2, width=224 * 2, scale=(0.5745, 1.0), ratio=(0.75, 1.33), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),#EfficientNet and ResNet50
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=min_visibility))


        self.pretrain_query_Transform = A.Compose([
            A.Resize(128, 128),
            A.HorizontalFlip(p=0.5),  # Optional: Add horizontal flip with 50% probability
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.ToGray(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # EfficientNet Resnet50
            A.pytorch.ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=min_visibility))


    # ‘pascal_voc’: [x_min, y_min, x_max, y_max]
    # ‘albumentations’: [x_min, y_min, x_max, y_max] normalized
    # ‘coco’ - [x_min, y_min, width, height]
    # `‘yolo’: [x_center, y_center, width, height] normalized
    def __call__(self, image, bboxes,format,ImgType):


        # Convert bboxes to pascal_voc format:  [x_min, y_min, x_max, y_max] -  here the cropping magic happens
        aug_bboxes=[]
        for bbox in bboxes:
            if format == 'coco':
                aug_bboxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]+['1'])

            if format == 'pascal_voc':
                aug_bboxes.append(bbox + ['1'])


        if self.Mode == 'PreTrain' or self.Mode == 'Train':
            TargetTransform = self.pretrain_target_transform
            QueryTransform  = self.pretrain_query_Transform

        if self.Mode == 'Test':
            TargetTransform = self.zero_target_transform
            QueryTransform  = self.zero_query_Transform

        # Apply augmentation
        k=0
        while True:
            try:
                if ImgType == 'Target':
                    augmented =TargetTransform(image=image, bboxes=aug_bboxes)
                    augmented_image = augmented['image']
                    augmented_bboxes = np.array(augmented['bboxes'][0][:-1])
                    break
            except:
                if False:
                    augmented_image = np.transpose(augmented_image.numpy(), (2, 1, 0)).astype(np.uint8)  # RGB->BGR
                    cv2.rectangle(augmented_image, (int(augmented_bboxes[0]), int(augmented_bboxes[1])),
                                  (int(augmented_bboxes[2]), int(augmented_bboxes[3])),
                                  (0, 255, 0), 2)
                    cv2.imshow("Image1", augmented_image)
                    cv2.waitKey(0)
                k += 1
                if k > 5:

                    try:
                        augmented = self.zero_target_transform(image=image, bboxes=aug_bboxes)
                        augmented_image = augmented['image']
                        augmented_bboxes = np.array(augmented['bboxes'][0][:-1])
                        break
                    except:
                        aa=0

            if ImgType == 'Query':
                augmented = QueryTransform(image=image, bboxes=aug_bboxes)
                augmented_image = augmented['image']

                if self.Mode == 'Train' or self.Mode == 'Test':
                    augmented_bboxes = np.array(augmented['bboxes'][0][:-1])
                else:
                    augmented_bboxes=[]
                break


        # Convert bboxes back to PyTorch format
        augmented_bboxes = torch.tensor(augmented_bboxes, dtype=torch.float32)
        return augmented_image, augmented_bboxes



class CocoDetectionDataset(Dataset):
    def __init__(self, root, annFile, transform,Mode,min_w,min_h):
        self.coco = COCO(annFile)
        self.root = root
        self.transform = transform
        self.Mode = Mode
        self.min_w = min_w
        self.min_h = min_h


        self.categories = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.categories_list = list(self.categories.values())

        self.category_2_id = {self.categories[cat_key]: cat_key for cat_key in self.categories.keys()}

        #loop on all images an databses
        self.images_per_category = {cat_name: [] for cat_name in self.categories.values()}
        for img_id in self.coco.getImgIds():
            #get annotaion IDs PER image
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                cat_id = ann['category_id']
                cat_name = self.categories[cat_id]
                self.images_per_category[cat_name].append(img_id)

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, index):
        k=0
        while True:

            try:
                k+=1
                #if k>5:print('k='+repr(k))

                # Select a category randomly
                category_name = self.categories_list[randint(0, len(self.categories_list)-1)]
                category_id = self.category_2_id[category_name]
                img_ids = self.images_per_category[category_name]

                # Select two images from the same category

                idx = random.sample(range(len(img_ids)), 2)
                img_id1 = img_ids[idx[0]]
                img_id2 = img_ids[idx[1]]


                # Get the annotations for the images - the images might contain objects from multiple categories
                ann_ids1 = self.coco.getAnnIds(imgIds=img_id1,areaRng=[self.min_w*self.min_h,1e10], catIds=[category_id])
                anns1 = self.coco.loadAnns(ann_ids1)
                if len(anns1) == 0:continue

                ann1 = anns1[np.random.randint(len(anns1))]

                # ‘coco’ - [x_min, y_min, width, height]
                bbox1 = [math.floor(num) for num in ann1['bbox']]
                if bbox1[3] < self.min_h or bbox1[2] < self.min_w:continue

                if self.Mode == 'PreTrain':break


                ann_ids2 = self.coco.getAnnIds(imgIds=img_id2,areaRng=[self.min_w*self.min_h,1e10], catIds=[category_id])
                anns2 = self.coco.loadAnns(ann_ids2)
                if len(anns2) == 0: continue




                ann2 = anns2[np.random.randint(len(anns2))]
                # ‘coco’ - [x_min, y_min, width, height]
                bbox2 =  [math.floor(num) for num in ann2['bbox']]

                if (bbox2[3] > self.min_h) and (bbox2[2] > self.min_w):
                    break
            except:
                print('Error in loader')

        # Load the images
        img1 = Image.open(os.path.join(self.root, self.coco.loadImgs(img_id1)[0]['file_name'])).convert("RGB")
        img1 = np.array(img1)

        if self.Mode != 'PreTrain':
            QueryImg = Image.open(os.path.join(self.root, self.coco.loadImgs(img_id2)[0]['file_name'])).convert("RGB")
            QueryImg = np.array(QueryImg)

        # plt.imshow(img1);plt.show()
        # plt.imshow(img2);plt.show()

        if False:
            image = img1
            cv2.rectangle(image, (bbox1[0], bbox1[1]), (bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]),(0, 255, 0), 2)
            cv2.imshow("Image1", image[:, :, [2, 1, 0]])  # RGB->BGR
            key = cv2.waitKey(0)

            image = QueryImg
            cv2.rectangle(image,  (bbox2[0], bbox2[1]), (bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]),(0, 255, 0), 2)
            cv2.imshow("Image1", image[:, :, [2, 1, 0]])  # RGB->BGR
            key = cv2.waitKey(0)


        #while True:
        TargetImg_out,TargetBbox_out = self.transform(img1,[bbox1],format='coco',ImgType = 'Target')

        if self.Mode == 'PreTrain' or self.Mode == 'PreTrainTest':
            #use the ROI from the Target image as a Query

            try:
                QueryImg = img1[bbox1[1]:(bbox1[1]+bbox1[3]),bbox1[0]:(bbox1[0]+bbox1[2]),:]
                QueryImg_out,QueryBbox_out = self.transform(QueryImg,[],format='coco',ImgType = 'Query')
            except:
                QueryImg_out, QueryBbox_out = self.transform(QueryImg, [], format='coco', ImgType='Query')
                aa=9
        else:
            QueryImg_out, QueryBbox_out = self.transform(QueryImg, [bbox2], format='coco', ImgType='Query')

        TargetImg  = TargetImg_out
        TargetBbox = TargetBbox_out

        QueryImg  = QueryImg_out
        QueryBbox =QueryBbox_out



        if False:
            bbox1     = TargetBbox.int()
            image = TargetImg_out.permute(1, 2, 0).numpy().astype(np.uint8)
            cv2.rectangle(image, (bbox1[0].item(), bbox1[1].item()), (bbox1[2].item(), bbox1[3].item()), (0, 255, 0), 2)
            cv2.imshow("Image1", image[:, :, [2, 1, 0]])# RGB->BGR
            key = cv2.waitKey(0)

            QueryBbox = QueryBbox.int()
            image = QueryImg.permute(1, 2, 0).numpy().astype(np.uint8)
            if self.Mode != 'PreTrain':
                cv2.rectangle(image, (QueryBbox[0].item(), QueryBbox[1].item()), (QueryBbox[2].item(), QueryBbox[3].item()),(0, 255, 0), 2)
            cv2.imshow("Image2", image[:, :, [2, 1, 0]])# RGB->BGR
            key = cv2.waitKey(0)
            # cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)

        TargetBbox     = TargetBbox/torch.Tensor([TargetImg.shape[2], TargetImg.shape[1],TargetImg.shape[2], TargetImg.shape[1]])
        #QueryBbox =torch.Tensor(QueryBbox)/torch.Tensor([QueryImg.shape[2],QueryImg.shape[1],QueryImg.shape[2], QueryImg.shape[1]])

        # Return the whole image, cropped object, and ROI
        return TargetImg.float(), QueryImg.float(), TargetBbox,QueryBbox




class ImageNet100Dataset(Dataset):
    def __init__(self, root, annFile, transform,Mode,min_w,min_h):
        self.coco = COCO(annFile)
        self.root = root
        self.transform = transform
        self.Mode = Mode
        self.min_w = min_w
        self.min_h = min_h


        self.categories = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.categories_list = list(self.categories.values())

        self.category_2_id = {self.categories[cat_key]: cat_key for cat_key in self.categories.keys()}

        #loop on all images an databses
        self.images_per_category = {cat_name: [] for cat_name in self.categories.values()}
        for img_id in self.coco.getImgIds():
            #get annotaion IDs PER image
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                cat_id = ann['category_id']
                cat_name = self.categories[cat_id]
                self.images_per_category[cat_name].append(img_id)

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, index):
        k=0
        while True:

            try:
                k+=1
                #if k>5:print('k='+repr(k))

                # Select a category randomly
                category_name = self.categories_list[randint(0, len(self.categories_list)-1)]
                category_id = self.category_2_id[category_name]
                img_ids = self.images_per_category[category_name]

                # Select two images from the same category

                idx = random.sample(range(len(img_ids)), 2)
                img_id1 = img_ids[idx[0]]
                img_id2 = img_ids[idx[1]]


                # Get the annotations for the images - the images might contain objects from multiple categories
                ann_ids1 = self.coco.getAnnIds(imgIds=img_id1,areaRng=[self.min_w*self.min_h,1e10], catIds=[category_id])
                anns1 = self.coco.loadAnns(ann_ids1)
                if len(anns1) == 0:continue

                ann1 = anns1[np.random.randint(len(anns1))]

                # ‘coco’ - [x_min, y_min, width, height]
                bbox1 = [math.floor(num) for num in ann1['bbox']]
                if bbox1[3] < self.min_h or bbox1[2] < self.min_w:continue

                if self.Mode == 'PreTrain':break


                ann_ids2 = self.coco.getAnnIds(imgIds=img_id2,areaRng=[self.min_w*self.min_h,1e10], catIds=[category_id])
                anns2 = self.coco.loadAnns(ann_ids2)
                if len(anns2) == 0: continue




                ann2 = anns2[np.random.randint(len(anns2))]
                # ‘coco’ - [x_min, y_min, width, height]
                bbox2 =  [math.floor(num) for num in ann2['bbox']]

                if (bbox2[3] > self.min_h) and (bbox2[2] > self.min_w):
                    break
            except:
                print('Error in loader')

        # Load the images
        img1 = Image.open(os.path.join(self.root, self.coco.loadImgs(img_id1)[0]['file_name'])).convert("RGB")
        img1 = np.array(img1)

        if self.Mode != 'PreTrain':
            QueryImg = Image.open(os.path.join(self.root, self.coco.loadImgs(img_id2)[0]['file_name'])).convert("RGB")
            QueryImg = np.array(QueryImg)

        # plt.imshow(img1);plt.show()
        # plt.imshow(img2);plt.show()

        if False:
            image = img1
            cv2.rectangle(image, (bbox1[0], bbox1[1]), (bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]),(0, 255, 0), 2)
            cv2.imshow("Image1", image[:, :, [2, 1, 0]])  # RGB->BGR
            key = cv2.waitKey(0)

            image = QueryImg
            cv2.rectangle(image,  (bbox2[0], bbox2[1]), (bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]),(0, 255, 0), 2)
            cv2.imshow("Image1", image[:, :, [2, 1, 0]])  # RGB->BGR
            key = cv2.waitKey(0)


        #while True:
        TargetImg_out,TargetBbox_out = self.transform(img1,[bbox1],format='coco',ImgType = 'Target')

        if self.Mode == 'PreTrain':
            #use the ROI from the Target image as a Query

            try:
                QueryImg = img1[bbox1[1]:(bbox1[1]+bbox1[3]),bbox1[0]:(bbox1[0]+bbox1[2]),:]
                QueryImg_out,QueryBbox_out = self.transform(QueryImg,[],format='coco',ImgType = 'Query')
            except:
                QueryImg_out, QueryBbox_out = self.transform(QueryImg, [], format='coco', ImgType='Query')
                aa=9
        else:
            QueryImg_out, QueryBbox_out = self.transform(QueryImg, [bbox2], format='coco', ImgType='Query')

        TargetImg  = TargetImg_out
        TargetBbox = TargetBbox_out

        QueryImg  = QueryImg_out
        QueryBbox =QueryBbox_out



        if False:
            bbox1     = TargetBbox.int()
            image = TargetImg_out.permute(1, 2, 0).numpy().astype(np.uint8)
            cv2.rectangle(image, (bbox1[0].item(), bbox1[1].item()), (bbox1[2].item(), bbox1[3].item()), (0, 255, 0), 2)
            cv2.imshow("Image1", image[:, :, [2, 1, 0]])# RGB->BGR
            key = cv2.waitKey(0)

            QueryBbox = QueryBbox.int()
            image = QueryImg.permute(1, 2, 0).numpy().astype(np.uint8)
            if self.Mode != 'PreTrain':
                cv2.rectangle(image, (QueryBbox[0].item(), QueryBbox[1].item()), (QueryBbox[2].item(), QueryBbox[3].item()),(0, 255, 0), 2)
            cv2.imshow("Image2", image[:, :, [2, 1, 0]])# RGB->BGR
            key = cv2.waitKey(0)
            # cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)

        TargetBbox     = TargetBbox/torch.Tensor([TargetImg.shape[2], TargetImg.shape[1],TargetImg.shape[2], TargetImg.shape[1]])
        #QueryBbox =torch.Tensor(QueryBbox)/torch.Tensor([QueryImg.shape[2],QueryImg.shape[1],QueryImg.shape[2], QueryImg.shape[1]])

        # Return the whole image, cropped object, and ROI
        return TargetImg.float(), QueryImg.float(), TargetBbox,QueryBbox









class WarmUpScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, warmup_factor, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.Finished = False
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def IsFinished(self):
        return self.Finished

    def get_lr(self):

        if self.Finished:
            Result = []
            for param_group in self.optimizer.param_groups:
                Result.append(param_group['lr'])
            return Result

        if self.last_epoch == self.warmup_steps:
            self.Finished = True

        if self.last_epoch <= self.warmup_steps:
            return [(base_lr / self.warmup_factor + (base_lr - base_lr / self.warmup_factor) * (
                        self.last_epoch / self.warmup_steps)) for base_lr in self.base_lrs]

def GetSystemData():
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__pyTorch CUDA VERSION:', torch.version.cuda)
    print("Torchvision version:", torchvision.__version__)
    print("Is CUDA enabled?", torch.cuda.is_available())
    print('__CUDA VERSION')
    # subprocess.call(["nvcc", "--version"])
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    subprocess.call(["nvidia-smi", "--format=csv",
                     "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())

# Function to filter the dataset to a given percentage
def filter_dataset(dataset, percentage):
    # Determine the number of samples to use
    total_samples = len(dataset)
    subset_size = int(total_samples * (percentage / 100))
    # Randomly sample a subset of indices
    indices = random.sample(range(total_samples), subset_size)
    # Return a subset of the dataset
    return Subset(dataset, indices)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self,dataset,percentage=None, *args, **kwargs):
        if percentage:
            dataset = filter_dataset(dataset, percentage)
        super().__init__(dataset,*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

        self.Mode = "Default"

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def GetNativeSamples(self,idx):
        CurrentSamplingMode =  self.dataset.SamplingMode
        self.SamplingMode   = 'Native'

        batch_size = self.dataset.batch_size
        self.dataset.batch_size = idx.shape[0]

        if Mode:
            CurrentTransformMode = self.dataset.TransformMode
            self.dataset.TransformMode   = TransformMode

        Result = self.__getitem__(idx)

        self.dataset.SamplingMode = CurrentSamplingMode
        self.dataset.batch_size = batch_size

        return Result


    def __iter__(self):

        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)



class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

class InfiniteSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        while True:
            for sample in self.data_source:
                yield sample




def LoadModel(net,StartBestModel,ModelsDirName,BestFileName,UseBestScore,device,prefix):

    scheduler = None
    optimizer = None
    WarmUpSched = None

    HighestACC = -1e5

    if StartBestModel:
        FileList = glob.glob(ModelsDirName + BestFileName + '.pth')
    else:
        FileList = glob.glob(ModelsDirName + prefix + "*pth")

    if len(FileList) == 0:
        FileList = glob.glob(ModelsDirName + prefix + "*pth")

    if FileList:
        FileList.sort(key=os.path.getmtime)

        OutputStr = FileList[-1] + ' loded'

        checkpoint = torch.load(FileList[-1])

        if UseBestScore:
            HighestACC = checkpoint['HighestACC']

        #get input network data
        net_dict = net.state_dict()
        checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if
                                    (k in net_dict) and (net_dict[k].shape == checkpoint['state_dict'][k].shape)}

        net.load_state_dict(checkpoint['state_dict'], strict=False)

        if 'optimizer' in checkpoint.keys():
            try:
                optimizer0 = copy.deepcopy(checkpoint['optimizer'])
                optimizer  = copy.deepcopy(checkpoint['optimizer'])
            except Exception as e:
                print(e)
                print('Optimizer loading error')

        WarmUpSched = None
        if 'warmup_scheduler_name' in checkpoint.keys():
            WarmUpSched = WarmUpScheduler(optimizer0, warmup_steps=10, warmup_factor=10)
            WarmUpSched.load_state_dict(checkpoint['warmup_scheduler'])
            WarmUpSched.optimizer = optimizer

        scheduler = None
        if 'scheduler_name' in checkpoint.keys():
            try:
                if checkpoint['scheduler_name'] == 'ReduceLROnPlateau':
                    scheduler = ReduceLROnPlateau(optimizer0, mode='min', factor=0.5, patience=6, verbose=True)
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    scheduler.optimizer=optimizer

                if checkpoint['scheduler_name'] == 'CosineAnnealingWarmRestarts':
                    scheduler = CosineAnnealingWarmRestarts(optimizer0, T_0=200, T_mult=1, eta_min=1e-5,last_epoch=-1)
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    scheduler.optimizer=optimizer

                if checkpoint['scheduler_name'] == 'StepLR':
                    scheduler = StepLR(optimizer0, step_size=1, gamma=10)
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    scheduler.optimizer = optimizer

                if checkpoint['scheduler_name'] == 'CosineAnnealingLR':
                    scheduler = CosineAnnealingLR(optimizer0, T_max=int(1.5e5), eta_min=0.)
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    scheduler.optimizer = optimizer


                if checkpoint['scheduler_name'] == 'GradualWarmupSchedulerV2':
                    Dict = copy.deepcopy(checkpoint['scheduler'])

                    if type(Dict['after_scheduler'])== torch.optim.lr_scheduler.ReduceLROnPlateau:
                        scheduler = GradualWarmupSchedulerV2(copy.deepcopy(optimizer), multiplier=1, total_epoch=10,after_scheduler=None)

                        scheduler.load_state_dict(checkpoint['scheduler'])
                        scheduler.after_scheduler=Dict['after_scheduler']

                        optimizer = scheduler.after_scheduler.optimizer


            except Exception as e:
                print(e)
                print('Scheduler loading error')

        StartEpoch = checkpoint['epoch'] + 1
    else:
        OutputStr = 'Weights file NOT loaded!!'
        optimizer  = None
        StartEpoch = 0

    print(OutputStr)

    FileList = glob.glob(ModelsDirName + BestFileName + '.pth')
    # noinspection PyInterpreter

    if FileList:
        checkpoint = torch.load(FileList[-1])

        if UseBestScore:
            HighestACC = checkpoint['HighestACC']


    print('HighestACC: ' + repr(HighestACC)[0:6])

    return net,optimizer,HighestACC,StartEpoch,scheduler,WarmUpSched,OutputStr



def create_mosaic(images, probabilities=[0.5, 0.33, 0.17]):
    # Select mosaic type based on probabilities
    mosaic_type = np.random.choice([1, 2, 3], p=probabilities)

    # If mosaic_type is 1, just return the original image
    if mosaic_type == 1:
        return random.choice(images)

    # Create an empty array for the mosaic
    mosaic_image = np.zeros((images[0].shape[0]*mosaic_type, images[0].shape[1]*mosaic_type, images[0].shape[2]))

    # Fill the mosaic with images
    for i in range(mosaic_type):
        for j in range(mosaic_type):
            # Select a random image
            image = random.choice(images)
            # Place the image in the mosaic
            mosaic_image[i*image.shape[0]:(i+1)*image.shape[0], j*image.shape[1]:(j+1)*image.shape[1]] = image

    return mosaic_image