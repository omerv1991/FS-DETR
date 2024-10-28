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
from torch.nn import TransformerDecoderLayer,TransformerDecoder
import timm
from torchvision import models
from transformers import CLIPModel


import cv2


import torch




#(batchsize, x, y, ch)
class PositionalEncoding2D(nn.Module):
    def __init__(self,channels_no,device=None,max_width=256,max_height=256):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels_no = int(np.ceil(channels_no/2))
        self.channels_no = channels_no
        inv_freq = 1. / (10000 ** (torch.arange(0, channels_no, 2).float() / channels_no))
        self.register_buffer('inv_freq', inv_freq)

        self.max_width  = max_width
        self.max_height = max_height
        self.device     = device

        self.pe2d = self.compute_pe2D(device,max_width,max_height)

    def compute_pe2D(self,device,max_width,max_height):

        pos_x = torch.arange(max_width, device=device).type(self.inv_freq.type())
        pos_y = torch.arange(max_height, device=device).type(self.inv_freq.type())

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)

        emb = torch.zeros((max_width, max_height, self.channels_no * 2), device=device)

        # concatenate embeddings
        emb[:, :, :self.channels_no] = emb_x
        emb[:, :, self.channels_no:(2*self.channels_no)] = emb_y

        return emb


    def forward(self, x):
        #return x+self.pe2d[None,:,:,:orig_ch]
        #fixme what is it for?
        return x+self.pe2d[None,:,:,:self.channels_no]



class FewShotNetwork(nn.Module):
    def __init__(self,p=0.2,VisualPrompt_flag=False,object_embeddings_size=5):
        super(FewShotNetwork, self).__init__()
        # Load a pre-trained MViT model from the timm library
        #print(timm.list_models())
        #self.backbone = timm.create_model('beit_base_patch16_224', pretrained=True)
        #self.backbone  = models.efficientnet_b5(pretrained=True)
        self.backbone  = MyResNet50()
        #self.backbone  = models.resnet50(pretrained=True)
        #self.backbone  =CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        #self.backbone = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        #self.backbone  = self.backbone.features[:-5]
        #self.backbone = self.backbone.features[:-3]
        self.object_embeddings_size = object_embeddings_size
        d_model=1024
        self.d_model = d_model
        self.LayerNorm1 = nn.LayerNorm(d_model)
        self.LayerNorm2 = nn.LayerNorm(d_model)
        self.LayerNorm3 = nn.LayerNorm(d_model)
        self.LayerNorm4 = nn.LayerNorm(d_model)


        #d_model=176
        Poss2D=PositionalEncoding2D(channels_no=d_model, max_height=256, max_width=256)
        self.PositionalEncoding2D = Poss2D.pe2d.permute(2, 0, 1)
        self.HEAD_embedding = nn.Embedding(num_embeddings=self.object_embeddings_size, embedding_dim=d_model)
        self.HEAD_embedding = nn.Parameter(F.normalize(torch.randn(self.object_embeddings_size, d_model), dim=1))
        k_shot = 1
        query_dim = 64
        self.pseudo_class_embedding = nn.Parameter(F.normalize(torch.randn(k_shot,query_dim, d_model), dim=1))

        nhead = 2
        num_layers = 2
        dim_feedforward = 1024

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1)
        self.transformer_decoder_4DCV = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1)
        self.transformer_decoder_detection_head = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        nhead = 4
        num_layers = 4
        dim_feedforward = 1024
        self.VisualPrompt_flag = VisualPrompt_flag
        #decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1)
        #self.transformer_decoder_regression_head = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.Class_head1 = nn.Linear(d_model, d_model)
        self.Class_head2 = nn.Linear(d_model, d_model)
        self.Class_head3 = nn.Linear(d_model, 1)

        self.BBOX_head1 = nn.Linear(d_model, d_model)
        self.BBOX_head2 = nn.Linear(d_model, d_model)
        self.BBOX_head3 = nn.Linear(d_model, 1)
        # p = dropout rate
        self.dropout1 = nn.Dropout(p=p)
        self.dropout2 = nn.Dropout(p=p)
        self.dropout3 = nn.Dropout(p=p)

    def FreezeBackbone(self,Flag):
        for param in self.backbone.parameters():
            param.requires_grad = not Flag

    def forward(self, TargetImg,QueryImg):

        if TargetImg.isnan().any():
            print("nan in TargetImg\n")
            sys.exit()
        if QueryImg.isnan().any():
            print("nan in QueryImg\n")
            sys.exit()

            # Pass the input through the backbone
        Target = self.backbone(TargetImg).permute(0, 2, 3, 1)
        Query = self.backbone(QueryImg).permute(0, 2, 3, 1)

        if Target.isnan().any() or self.LayerNorm1.bias.isnan().any():
            print("nan in Target0\n")

        if Query.isnan().any() or self.LayerNorm2.bias.isnan().any():
            print("nan in query0\n")

        Target = self.LayerNorm1(Target).permute(0, 3, 1, 2)
        Query = self.LayerNorm2(Query).permute(0, 3, 1, 2)

        if Target.isnan().any():
            print("nan in Target1\n")
            sys.exit()
        if Query.isnan().any():
            print("nan in query1\n")
            sys.exit()

        # prepare positional encoding
        TargetPE = self.PositionalEncoding2D[:, :Target.shape[2], :Target.shape[3]]
        QueryPE = self.PositionalEncoding2D[:, :Query.shape[2], :Query.shape[3]]

        TargetPE = TargetPE.unsqueeze(0).repeat(Target.shape[0], 1, 1, 1).to(Target.device)
        QueryPE = QueryPE.unsqueeze(0).repeat(Query.shape[0], 1, 1, 1).to(Target.device)

        # Target += TargetPE
        # Query  += QueryPE

        # Reshape the tensor to [B, C, H*W] row-wise
        Target = Target.reshape(Target.size(0), Target.size(1), -1).permute(2, 0, 1)
        Query = Query.reshape(Query.size(0), Query.size(1), -1).permute(2, 0, 1)

        TargetPE = TargetPE.reshape(TargetPE.size(0), TargetPE.size(1), -1).permute(2, 0, 1)
        #fixme should i use is? what for?
        QueryPE = QueryPE.reshape(QueryPE.size(0), QueryPE.size(1), -1).permute(2, 0, 1)

        # Target = self.transformer_encoder(src=Target)
        # Query  = self.transformer_encoder(src=Query, src_key_padding_mask=memory_key_padding_mask)

        if self.VisualPrompt_flag:
            # use the pseudo class embedding
            batch_size = Query.shape[1]
            pseudo_class_embedding = self.pseudo_class_embedding.unsqueeze(1).repeat(1,batch_size,1, 1).permute(0,2,1,3)
            # in case k=1 -> else- use .repeat?
            pseudo_class_embedding = pseudo_class_embedding.squeeze(0).to(Query.device)
            # create the visual prompt
            visual_prompt = Query + pseudo_class_embedding
            TransTarget = self.transformer_decoder_4DCV(tgt=Target, memory=visual_prompt)

        else:
            TransTarget = self.transformer_decoder_4DCV(tgt=Target, memory=Query)

        if TransTarget.isnan().any():
            print("nan in TransTarget")

        # prepare negative results
        ShuffledTarget = Target[:, torch.randperm(Target.size(1)), :]
        NegativeTransTarget = self.transformer_decoder_4DCV(tgt=ShuffledTarget, memory=Query)

        TransTarget += TargetPE
        NegativeTransTarget += TargetPE

        self.HEAD_embedding.data = F.normalize(self.HEAD_embedding.data, dim=1)
        HeadQuery = self.HEAD_embedding.unsqueeze(1).repeat(1, TransTarget.shape[1], 1, )

        if self.VisualPrompt_flag:
            pass
            # concat with the object query ; HeadQuery is the object_queries
            concatenated_queries = torch.cat((HeadQuery, visual_prompt), dim=0)
            # fixme since trans target has only 5 queries (1st for classify and rest for bbox) PCE be implemented!
            #self.object_embeddings_size = 69
            #HeadQuery = concatenated_queries

        PosResultsDetect = self.transformer_decoder_detection_head(tgt=HeadQuery[0:self.object_embeddings_size, :], memory=TransTarget)
        NegResultsDetect = self.transformer_decoder_detection_head(tgt=HeadQuery[0:self.object_embeddings_size, :], memory=NegativeTransTarget)

        PosResultsDetect = self.LayerNorm3(PosResultsDetect)
        NegResultsDetect = self.LayerNorm3(NegResultsDetect)

        # PosResultsRegress = self.transformer_decoder_regression_head(tgt=HeadQuery[1:2, :], memory=TransTarget)
        # PosResultsRegress = self.LayerNorm4(PosResultsRegress)

        if PosResultsDetect.isnan().any():
            print("PosResultsDetect nan")
            sys.exit()

        if NegResultsDetect.isnan().any():
            print("NegResultsDetect nan")
            sys.exit()

        # output MLP
        PosClass = self.Class_head1(PosResultsDetect[0,]).squeeze()
        PosClass = torch.nn.functional.relu(PosClass)
        PosClass = self.dropout1(PosClass)
        PosClass = self.Class_head3(PosClass).squeeze()

        NegClass = self.Class_head1(NegResultsDetect[0,]).squeeze()
        NegClass = torch.nn.functional.relu(NegClass)
        NegClass = self.dropout2(NegClass)
        NegClass = self.Class_head3(NegClass).squeeze()

        PosBbox = self.BBOX_head1(PosResultsDetect[1:self.object_embeddings_size, ]).squeeze()

        # PosBbox = self.BBOX_head1(PosResultsRegress).squeeze()
        PosBbox = torch.nn.functional.relu(PosBbox)
        # PosBbox = self.BBOX_head2(PosBbox).squeeze()
        # PosBbox = torch.nn.functional.relu(PosBbox)
        PosBbox = self.dropout3(PosBbox)
        PosBbox = self.BBOX_head3(PosBbox).squeeze().permute(1, 0)
        # PosBbox = torch.nn.functional.relu(PosBbox)
        PosBbox = PosBbox.clamp(min=0, max=1)

        if PosClass.isnan().any(): print("NAN PosClass")
        if NegClass.isnan().any(): print("NAN NegClass")
        if PosBbox.isnan().any(): print("NAN PosBbox")

        return PosClass, PosBbox, NegClass

class MyResNet50(models.ResNet):
    def __init__(self):
        super(MyResNet50, self).__init__(models.resnet.Bottleneck, [3, 4, 6, 3])
        self.load_state_dict(models.resnet50(pretrained=True).state_dict())

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        #return x

        layer3_output = self.layer3(x)
        layer4_output = self.layer4(layer3_output)

        #x = self.avgpool(layer4_output)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return layer3_output





