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

from few_shots_utils_detection import (CocoDetectionDataset,WarmUpScheduler,GetSystemData,MultiEpochsDataLoader,LoadModel,compute_iou,generalized_box_iou_loss)
from few_shots_network import FewShotNetwork,ObjectDetectionAugmentation

# Define the root directory where images are stored and the path to the annotation file
TrainDir = 'F:\\coco\\train2017\\'
TrainAnnotations = 'F:\\coco\\annotations\\instances_train2017.json'

TestDir = 'F:\\coco\\val2017\\'
TestAnnotations = 'F:\\coco\\annotations\\instances_val2017.json'



if __name__ == '__main__':
    GetSystemData()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NumGpus = torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()

    ModelsDirName = './models1/'
    LogsDirName = './logs1/'
    Description = 'DeepAge'
    BestFileName = 'detection_best'
    FileName = 'detection'


    InitializeOptimizer = True

    batch_size = 32*NumGpus
    no_epochs  = 500

    LearningRate = 1e-5
    MinLearningRate = 1e-7
    DropoutP = 0.2
    weight_decay = 1e-4
    patience = 20
    WarmUpEpochs = 10
    ModelSaveInterval = 1

    IoU_T = 0.5
    min_w = 40
    min_h = 40

    W_L1 = 5
    W_gIoU = 2


    FreezeBackbone = True
    StartBestModel = True
    UseBestScore   = True

    # Create the dataset
    TrainAugmentation=ObjectDetectionAugmentation(p=0.5,min_visibility=0.6,Mode='Train')
    TestAugmentation = ObjectDetectionAugmentation(p=0.5,min_visibility=0.6,Mode='Test')

    TrainDataset = CocoDetectionDataset(TrainDir, TrainAnnotations, transform=TrainAugmentation,min_w=min_w,min_h=min_h)
    TestDataset  = CocoDetectionDataset(TestDir,  TestAnnotations,  transform=TestAugmentation,min_w=min_w,min_h=min_h)

    # Create a DataLoader for the dataset
    TrainDataLoader = MultiEpochsDataLoader(TrainDataset, batch_size=batch_size,shuffle=True, num_workers=6)
    TestDataLoader  = MultiEpochsDataLoader(TestDataset,  batch_size=batch_size,shuffle=False, num_workers=8)

    bce_loss = nn.BCEWithLogitsLoss()

    net = FewShotNetwork()

    StartEpoch = 0
    HighestACC = -1e10
    net, optimizer, HighestACC, StartEpoch, scheduler, WarmUpSched, OutputStr = (
        LoadModel(net, StartBestModel, ModelsDirName, BestFileName, UseBestScore, device, prefix=FileName))

    CurrentError = HighestACC

    net.FreezeBackbone(FreezeBackbone)

    net.to(device)
    if NumGpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    if InitializeOptimizer:
        optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad == True, net.parameters()), 'lr': LearningRate,
              'weight_decay': weight_decay},
             {'params': filter(lambda p: p.requires_grad == False, net.parameters()), 'lr': 0, 'weight_decay': 0}],
            lr=0, weight_decay=0.00)

        WarmUpSched = WarmUpScheduler(optimizer, warmup_steps=WarmUpEpochs, warmup_factor=10)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)



    for k in range(2*StartEpoch,2*no_epochs):  # Number of epochs
        epoch = k//2

        if not WarmUpSched.IsFinished():
            print('\n' + Fore.BLACK + Back.RED + ' Warmup step #' + repr(WarmUpSched.last_epoch) + Style.RESET_ALL)
        else:
            print(repr(scheduler.patience - scheduler.num_bad_epochs) + ' epochs to rate reduction')

        # Print Learning Rates
        string = 'LR: '
        for param_group in optimizer.param_groups:
            string += repr(param_group['lr']) + ' '
        print(string)

        running_class_acc = 0
        short_running_class_acc = 0
        running_IoU = 0

        if k%2 == 0: Mode = 'Train'
        else:        Mode = 'Test'

        if Mode=='Train':
            bar = tqdm(TrainDataLoader, desc="Training", leave=True, position=0)
            net = net.train()
        if Mode=='Test':
            bar = tqdm(TestDataLoader, desc="Testing", leave=True, position=0)
            net = net.eval()

        TotalLoss = 0
        torch.set_grad_enabled(Mode == 'Train')
        for batch, (imgs1, QueryImg, bbox1,QueryBbox) in enumerate(bar):

            imgs1, QueryImg = imgs1.to(device, non_blocking=True), QueryImg.to(device, non_blocking=True)
            bbox1 = bbox1.to(device, non_blocking=True)


            PosClass,PosBbox,NegClass = net(imgs1,QueryImg,QueryBbox)

            if PosClass.isnan().any() or PosBbox.isnan().any() or NegClass.isnan().any():
                print("Output NAN")

            pos_class_loss  = bce_loss(PosClass, torch.ones(PosClass.shape[0]).to(PosClass.device))
            neg_class_loss  = bce_loss(NegClass, torch.zeros(NegClass.shape[0]).to(NegClass.device))

            running_class_acc += ((PosClass > 0).sum() + (NegClass<0).sum()).item()/(PosClass.shape[0]+NegClass.shape[0])
            short_running_class_acc += ((PosClass > 0).sum() + (NegClass<0).sum()).item()/(PosClass.shape[0]+NegClass.shape[0])


            #bbox1[:, 2] = bbox1[:, 2] - bbox1[:, 0]  # w
            #bbox1[:, 3] = bbox1[:, 3] - bbox1[:, 1]  # h

            bbox_loss = (PosBbox-bbox1).abs().mean()
            gIoU = generalized_box_iou_loss(PosBbox,bbox1)
            #L1 = 5 and gIoU = 2

            box_loss = W_L1*bbox_loss+W_gIoU*gIoU
            bbox_loss *= 1

            #loss = pos_class_loss+neg_class_loss+bbox_loss
            loss = bbox_loss



            #PosBbox[:, 2] = PosBbox[:, 2] + PosBbox[:, 0]  # w
            #PosBbox[:, 3] = PosBbox[:, 3] + PosBbox[:, 1]  #

            IoU = compute_iou(PosBbox,bbox1)
            IoU = (IoU>IoU_T).sum()/IoU.shape[0]
            running_IoU+=IoU

            TotalLoss+=loss


            if batch%50 ==0 and batch>0 and Mode == 'Train':
                print('\nLoss = '+repr(loss.item())[0:4]+
                      ' pos_class_loss= '+repr(pos_class_loss.item())[0:4] +
                      ' neg_class_loss= '+repr(neg_class_loss.item())[0:4] +
                      ' bbox_loss= '    + repr(bbox_loss.item())[0:6] +
                      ' IoU>IoU_T =  '  + repr(IoU.item())[0:5])
                print(f"Running Class Acc: {short_running_class_acc/50:.4f}")
                short_running_class_acc = 0

            # Backward and optimize
            if Mode == 'Train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
                optimizer.step()

        if Mode == 'Test':

            WarmUpSched.step()
            CurrentACC=running_class_acc/batch
            if CurrentACC > HighestACC:
                ss=0
            scheduler.step(metrics=TotalLoss.item())

            print(Back.RED +'Epoch: '+repr(epoch)+ Style.RESET_ALL)
            print(Back.CYAN +'Test Loss = ' + repr(TotalLoss.item())[0:4] +
                  ' Test pos_class_loss= ' + repr(pos_class_loss.item())[0:4] +
                  ' Test neg_class_loss= ' + repr(neg_class_loss.item())[0:4]+ Style.RESET_ALL)
            print(Back.CYAN +f"Test Running Class Acc: {CurrentACC:.4f}"+f" Test Running IoU>IoU_T: {running_IoU/batch:.4f}"+ Style.RESET_ALL)

            state = {'epoch': epoch,
                     'state_dict': net.state_dict() if (NumGpus == 1) else net.module.state_dict(),
                     'optimizer': optimizer,
                     'scheduler_name': type(scheduler).__name__,
                     'scheduler': scheduler.state_dict(),
                     'warmup_scheduler_name': type(WarmUpSched).__name__,
                     'warmup_scheduler': WarmUpSched.state_dict(),
                     'HighestACC': HighestACC,
                     'batch_size': batch_size,
                     'DropoutP': DropoutP,
                     'weight_decay': weight_decay}


            if CurrentACC > HighestACC:
                HighestACC = CurrentACC

                print(Back.GREEN + 'Best error found and saved: ' + repr(HighestACC)[0:5] + Style.RESET_ALL)
                filepath = ModelsDirName + BestFileName + '.pth'
                Parallel(n_jobs=1, backend='multiprocessing')([delayed(torch.save)(state, filepath)])

            if ((epoch-1) % ModelSaveInterval) == 0 and epoch>0:
                filepath = ModelsDirName + FileName + repr(epoch) + '.pth'
                Parallel(n_jobs=1, backend='multiprocessing')([delayed(torch.save)(state, filepath)])
                print('Saved checkpoint ' + filepath)
                sys.stdout.flush()