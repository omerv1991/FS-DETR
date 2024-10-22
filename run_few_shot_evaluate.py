import torch
import torch.nn as nn
from torchvision import transforms
import GPUtil
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from colorama import init, Fore, Back, Style
from joblib import Parallel, delayed
import math
import sys
from few_shots_network import FewShotNetwork
from few_shots_utils_detection import (CocoDetectionDataset,ObjectDetectionAugmentation,WarmUpScheduler,GetSystemData,
                                       MultiEpochsDataLoader,InfiniteDataLoader,LoadModel,compute_iou)

# Define the root directory where images are stored and the path to the annotation file
TestDir = 'F:\\coco\\val2017\\'
TestAnnotations = 'F:\\coco\\annotations\\instances_val2017.json'

if __name__ == '__main__':
    GetSystemData()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NumGpus = torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()

    ModelsDirName = './models1/'
    LogsDirName = './logs3/'
    Description = 'DeepAge'
    #BestFileName = 'detection_best'
    #BestFileName = 'detection_best73'
    #BestFileName = 'detection_best110'
    BestFileName = 'detection_best46'
    FileName = 'detection'

    batch_size = 32*NumGpus
    no_epochs  = 100

    LearningRate = 1e-4
    MinLearningRate = 1e-7
    DropoutP = 0.2
    weight_decay = 1e-4
    patience = 20
    WarmUpEpochs = 10
    ModelSaveInterval = 10

    StartBestModel = True
    UseBestScore   = True

    FreezeBackbone = True


    # Create the dataset
    TestAugmentation = ObjectDetectionAugmentation(p=0.5,min_visibility=0.5,Mode='Test')
    min_w = 40
    min_h = 40

    TestDataset  = CocoDetectionDataset(TestDir,  TestAnnotations,  transform=TestAugmentation,min_w=min_w,min_h=min_h,Mode='Test')


    # Create a DataLoader for the dataset
    TestDataLoader  = MultiEpochsDataLoader(TestDataset,  batch_size=2*batch_size, shuffle=False, num_workers=8)

    net = FewShotNetwork()

    StartEpoch = 0
    net, optimizer, LowestError, StartEpoch, scheduler, WarmUpSched, OutputStr = (
        LoadModel(net, StartBestModel, ModelsDirName, BestFileName, UseBestScore, device, prefix='face_age'))

    CurrentError = LowestError

    net.FreezeBackbone(FreezeBackbone)

    net.to(device)
    if NumGpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    running_class_acc=0
    bar = tqdm(TestDataLoader, desc="Testing", leave=True, position=0)
    net = net.eval()

    torch.set_grad_enabled(False)
    iou_list = []
    for batch, (imgs1, QueryImg, bbox1,QueryBbox)  in enumerate(bar):

        imgs1, QueryImg = imgs1.to(device, non_blocking=True), QueryImg.to(device, non_blocking=True)
        bbox1 = bbox1.to(device, non_blocking=True)

        PosClass, PosBbox, NegClass = net(imgs1, QueryImg)

        running_class_acc += ((PosClass > 0).sum() + (NegClass<0).sum()).item()/(PosClass.shape[0]+NegClass.shape[0])

        bbox_loss = ((PosBbox-bbox1)**2).mean()

        iOu =compute_iou(PosBbox,bbox1)

        iou_list.append(iOu)



    CurrentACC = running_class_acc/batch
    print(f"Test Running Class Acc: {CurrentACC:.4f}")
    print(f"Test Running bbox_loss: {bbox_loss:.4f}")
    import numpy as np
    lists = [tensor.tolist() for tensor in iou_list]
    flattened_list = [item for sublist in lists for item in sublist]
    iou_flattened_list = np.array(flattened_list)
    print(f"Test Running iOu mean : {np.mean(iou_flattened_list):.4f}")
    filtered_iou = iou_flattened_list[iou_flattened_list > 0.5]
    print(f"Test Running iOu>0.5 acc : {len(filtered_iou) / len(iou_flattened_list):.4f}")

    filtered_iou = iou_flattened_list[iou_flattened_list > 0.3]
    print(f"Test Running iOu>0.3 acc : {len(filtered_iou) / len(iou_flattened_list):.4f}")

    filtered_iou = iou_flattened_list[iou_flattened_list > 0.01]
    print(f"Test Running iOu>0.01 acc : {len(filtered_iou) / len(iou_flattened_list):.4f}")

    filtered_iou = iou_flattened_list[iou_flattened_list > 0.7]
    print(f"Test Running iOu>0.7 acc : {len(filtered_iou) / len(iou_flattened_list):.4f}")


    i=0
    TP = np.zeros(len(iou_list*128))
    FP = np.zeros(len(iou_list*128))
    iou_threshold=0.5
    for iou_batch in iou_list:
        for iou_val in iou_batch:
            iou_val = float(iou_val)
            if iou_val >= iou_threshold :
                TP[i] = 1  # True positive
            else:
                FP[i] = 1  # False positive
            # Calculate Precision and Recall
            i+=1
    cum_TP = np.cumsum(TP)
    cum_FP = np.cumsum(FP)
    precision = cum_TP / (cum_TP + cum_FP)
    recall = cum_TP / len(iou_list*128)

    # Compute AP: area under the precision-recall curve
    ap50 = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    print (ap50)
