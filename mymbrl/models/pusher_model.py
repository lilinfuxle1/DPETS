import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from mymbrl.utils import swish, get_affine_params
import random
import math

class PusherModel(nn.Module):  

    def __init__(self, ensemble_size, in_features=30, out_features=46, hidden_size=300,  
                 drop_prob=0.2, dropout_mask_nums=30, device="cpu"):
        super().__init__()

        self.fit_input = True
        self.dropout = False
        self.mask_batch_size = 1
        self.batch_size = 128

        self.num_nets = ensemble_size

        # 建议调低dropout，调试时可先关闭
        self.drop_prob = drop_prob  # 0.2 默认，也可以传0关闭
        self.dropout_mask_nums = dropout_mask_nums
        self.hidden_size = hidden_size

        self.hidden1_mask = None
        self.hidden2_mask = None
        self.hidden3_mask = None

        self.hidden_mask_indexs = None
        self.hidden1_mask_select = None
        self.hidden2_mask_select = None
        self.hidden3_mask_select = None

        self.in_features = in_features
        self.out_features = out_features

        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size, in_features, hidden_size)
        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size, hidden_size, hidden_size)
        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size, hidden_size, hidden_size)
        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size, hidden_size, out_features)

        # 关键修改：inputs_sigma 初始化为全1，避免除零
        self.inputs_mu = nn.Parameter(torch.zeros(in_features).to(device), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.ones(in_features).to(device), requires_grad=False)

        # 固定 max_logvar 和 min_logvar 不参与训练，避免数值漂移导致NaN
        self.register_buffer('max_logvar', torch.ones(1, out_features // 2, dtype=torch.float32).to(device) / 2.0)
        self.register_buffer('min_logvar', - torch.ones(1, out_features // 2, dtype=torch.float32).to(device) * 10.0)
    
    def compute_decays(self):
        lin0_decays = 0.0002 * (self.lin0_w **2).sum() / 2.0
        lin1_decays = 0.0005 * (self.lin1_w** 2).sum() / 2.0
        lin2_decays = 0.0005 * (self.lin2_w **2).sum() / 2.0
        lin3_decays = 0.001 * (self.lin3_w** 2).sum() / 2.0
        return lin0_decays + lin1_decays + lin2_decays + lin3_decays

    def forward(self, inputs, ret_logvar=False, open_dropout=True):
        if self.fit_input:
            # 避免除零异常
            inputs = (inputs - self.inputs_mu) / (self.inputs_sigma + 1e-6)
        
        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden1_mask_select
        
        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden2_mask_select
        
        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden3_mask_select

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b

        mean = inputs[:, :, :self.out_features // 2]
        logvar = inputs[:, :, self.out_features // 2:]

        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar
        return mean, torch.exp(logvar)
    
    #逻辑不变
    def select_mask(self, batch_size=None):
        if not self.dropout:
            return
        self.batch_size = batch_size or self.batch_size
        index_list = list(range(0, self.dropout_mask_nums))
        index_tile_list = index_list * math.ceil(self.batch_size / self.dropout_mask_nums)
        device = self.get_param_device()
        indexs = torch.tensor(random.sample(index_tile_list, self.batch_size), device=device)
        self.hidden1_mask_select = torch.index_select(self.hidden1_mask, 1, indexs)
        self.hidden2_mask_select = torch.index_select(self.hidden2_mask, 1, indexs)
        self.hidden3_mask_select = torch.index_select(self.hidden3_mask, 1, indexs)

    def sample_new_mask(self, dropout_mask_nums=None, num_particles=None):
        self.dropout = True
        drop_prob = self.drop_prob
        device = self.get_param_device()
        if dropout_mask_nums:
            self.dropout_mask_nums = dropout_mask_nums
        # 掩码维度适配300隐藏层
        self.hidden1_mask = torch.bernoulli(
            torch.ones(self.num_nets, self.dropout_mask_nums, self.hidden_size) * (1 - drop_prob)).to(device)
        self.hidden2_mask = torch.bernoulli(
            torch.ones(self.num_nets, self.dropout_mask_nums, self.hidden_size) * (1 - drop_prob)).to(device)
        self.hidden3_mask = torch.bernoulli(
            torch.ones(self.num_nets, self.dropout_mask_nums, self.hidden_size) * (1 - drop_prob)).to(device)
    
    def get_param_device(self):
        return next(self.parameters()).device