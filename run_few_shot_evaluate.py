import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import GPUtil
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from colorama import init, Fore, Back, Style
from joblib import Parallel, delayed
import math
import sys

from few_shots_utils_detection import (CocoDetectionDataset,FewShotNetwork,ObjectDetectionAugmentation,WarmUpScheduler,GetSystemData,
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
    BestFileName = 'detection_best'
    FileName = 'detection'

    batch_size = 384*NumGpus
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
    TestDataset  = CocoDetectionDataset(TestDir,  TestAnnotations,  transform=TestAugmentation)

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
    for batch, (imgs1, QueryImg, bbox1,QueryBbox)  in enumerate(bar):

        imgs1, QueryImg = imgs1.to(device, non_blocking=True), QueryImg.to(device, non_blocking=True)
        bbox1 = bbox1.to(device, non_blocking=True)

        PosClass,PosBbox,NegClass = net(imgs1,QueryImg,QueryBbox)

        running_class_acc += ((PosClass > 0).sum() + (NegClass<0).sum()).item()/(PosClass.shape[0]+NegClass.shape[0])

        bbox_loss = ((PosBbox-bbox1)**2).mean()

        iOu =compute_iou(PosBbox,bbox1)

    CurrentACC = running_class_acc/batch
    print(f"Test Running Class Acc: {CurrentACC:.4f}")


