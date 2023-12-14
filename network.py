import torch
import torch.nn as nn
import torch.distributions as distributions
import numpy as np
import math
from torch.nn.modules.activation import ReLU

from torch.nn.modules.normalization import LayerNorm

from mlpmixer import MixerBlock
from config import *


def pairwise_cosine_similarity(x, y):
    '''
    calculate self-pairwise cosine distance
    input:
    x: torch.FloatTensor [B,C,L,E']
    y: torch.FloatTensor [B,C,L,E']
    output:
    xcos_dist: torch.FloatTensor [B,C,L,L]
    '''
    x = x.detach()
    y = y.permute(0,1,3,2)
    dot = torch.matmul(x, y)
    x_dist = torch.norm(x, p=2, dim=3, keepdim=True)
    y_dist = torch.norm(y, p=2, dim=2, keepdim=True)
    dist = x_dist * y_dist
    cos = dot / (dist + 1e-8)
    return cos

def pairwise_minus_l2_distance(x, y):
    '''
    calculate pairwise l2 distance
    input:
    x: torch.FloatTensor [B,C,L,E']
    y: torch.FloatTensor [B,C,L,E']
    output:
    -l2_dist: torch.FloatTensor [B,C,L,L]
    '''
    x = x.unsqueeze(3).detach()
    y = y.unsqueeze(2)
    l2_dist = torch.sqrt(torch.sum((x-y)**2, dim=-1) + 1e-8)
    return -l2_dist



class CustomEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.feature_reduction = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.GELU()
        )
        self.short_layer = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 1),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 1),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 1),
        )
        self.middle_layer_1 = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=1),
        )
        self.middle_layer_2 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
        )
        #mixer
        self.mixer_layers = nn.ModuleList([MixerBlock(FEATURE_LEN, hidden, hidden//2, hidden*2) for _ in range(LONG_LAYER_NUM)])
        self.layernorm = nn.LayerNorm(hidden)
        

    def forward(self, x):
        '''
        IN: torch.FloatTensor [B,L,E]
        OUT: torch.FloatTensor [B,C,L,E']
        '''
        #first, reduce feature
        reduced_feature = self.feature_reduction(x).permute(0,2,1)
        #[B,E,L]
        short = self.short_layer(reduced_feature) + reduced_feature
        middle_1 = self.middle_layer_1(reduced_feature) + reduced_feature
        middle_2 = self.middle_layer_2(middle_1) + middle_1
        #do mixer computation
        long = reduced_feature.permute(0,2,1)
        #[B,L,E]
        for layer in self.mixer_layers:
            long = layer(long)
        long = self.layernorm(long)
        long = long.permute(0,2,1)
        out_list = [short, middle_1, middle_2, long]
        out = torch.stack(out_list, dim=1)
        assert CHANNEL_NUM % out.size(1) == 0
        group_num = CHANNEL_NUM // out.size(1)
        out = out.view(out.size(0), out.size(1), group_num, out.size(2)//group_num, out.size(3))
        out = out.view(out.size(0), -1, out.size(3), out.size(4)).permute(0,1,3,2)
        return out


# Temporary name for the network
class SJNET(nn.Module):
    def __init__(self, encoder_hidden=ENCODER_HIDDEN, channel_num=CHANNEL_NUM, decoder_hidden=DECODER_HIDDEN):
        super().__init__()
        self.encoder = CustomEncoder(encoder_hidden)
        self.normalize_tsm = nn.InstanceNorm2d(channel_num, affine=False)
        self.opt = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)
        
    def forward(self, x):
        #IN: [B, L, E]
        out = self.encoder(x)
        tsm = self.normalize_tsm(pairwise_cosine_similarity(out, out))
        return tsm

    def get_tsm(self, x):
        with torch.no_grad():
            out = self.encoder(x)
            tsm = self.normalize_tsm(pairwise_cosine_similarity(out, out))
            tsm = torch.mean(tsm, dim=1).unsqueeze(1)
            #for print, we permute x
            tsm = tsm.permute(0,2,3,1)
            tsm = tsm.detach().cpu().numpy()
        return tsm