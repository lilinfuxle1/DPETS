import torch
from torch import nn
from torch.nn import functional as F
import math
import random
from mymbrl.utils import swish, get_affine_params  # 确保你实现了这两个工具函数

class AntModel(nn.Module):
    """
    DPETS中Ant环境专用的概率动力学模型。
    输入维度：经过预处理后的状态（35维）+动作（8维）=43维
    输出维度：状态变化量（27维）均值和对数方差拼接为54维输出
    """

    def __init__(self, ensemble_size, in_features, out_features, hidden_size=200,
                 drop_prob=0.15, dropout_mask_nums=20, device="cpu"):
        super().__init__()

        self.num_nets = ensemble_size
        self.in_features = in_features
        self.out_features = out_features  # output mean + logvar, 所以是2倍状态维度

        self.drop_prob = drop_prob
        self.dropout_mask_nums = dropout_mask_nums
        self.hidden_size = hidden_size

        self.dropout = False
        self.hidden1_mask = None
        self.hidden2_mask = None
        self.hidden3_mask = None

        self.hidden1_mask_select = None
        self.hidden2_mask_select = None
        self.hidden3_mask_select = None

        # 使用论文中类似的权重结构，分别为ensemble中每个网络维护权重
        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size, in_features, hidden_size)
        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size, hidden_size, hidden_size)
        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size, hidden_size, hidden_size)
        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size, hidden_size, out_features)

        # 约束logvar范围，防止数值发散
        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2).to(device) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, out_features // 2).to(device) * 10.0)

        # 输入归一化参数，训练时需赋值
        self.inputs_mu = nn.Parameter(torch.zeros(in_features).to(device), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.ones(in_features).to(device), requires_grad=False)

        # batch size相关
        self.batch_size = 30

    def forward(self, inputs, ret_logvar=False, open_dropout=True):
        """
        inputs: [ensemble_size, batch_size, in_features]
        返回预测均值和方差，形状均为[ensemble_size, batch_size, out_features//2]
        """

        # 归一化输入
        inputs = (inputs - self.inputs_mu) / (self.inputs_sigma + 1e-8)

        # 线性层计算（手动乘权重+偏置）
        x = torch.bmm(inputs, self.lin0_w) + self.lin0_b  # [ens, batch, hidden]
        x = swish(x)
        if self.dropout and open_dropout:
            x = x * self.hidden1_mask_select

        x = torch.bmm(x, self.lin1_w) + self.lin1_b
        x = swish(x)
        if self.dropout and open_dropout:
            x = x * self.hidden2_mask_select

        x = torch.bmm(x, self.lin2_w) + self.lin2_b
        x = swish(x)
        if self.dropout and open_dropout:
            x = x * self.hidden3_mask_select

        x = torch.bmm(x, self.lin3_w) + self.lin3_b  # 输出层，形状[ens, batch, out_features]

        mean = x[:, :, :self.out_features // 2]
        logvar = x[:, :, self.out_features // 2:]

        # 约束logvar范围，提升数值稳定性
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def sample_new_mask(self, dropout_mask_nums=None):
        """ 采样新的dropout mask，固定mask直到下一次采样 """
        self.dropout = True
        device = self.get_param_device()
        if dropout_mask_nums is not None:
            self.dropout_mask_nums = dropout_mask_nums

        self.hidden1_mask = torch.bernoulli(
            torch.ones(self.num_nets, self.dropout_mask_nums, self.hidden_size) * (1 - self.drop_prob)).to(device)
        self.hidden2_mask = torch.bernoulli(
            torch.ones(self.num_nets, self.dropout_mask_nums, self.hidden_size) * (1 - self.drop_prob)).to(device)
        self.hidden3_mask = torch.bernoulli(
            torch.ones(self.num_nets, self.dropout_mask_nums, self.hidden_size) * (1 - self.drop_prob)).to(device)

    def select_mask(self, batch_size=None):
        """
        从已有的mask中选择batch_size个mask用于前向计算，保证训练时mask随机性和多样性
        """
        if not self.dropout:
            return
        if batch_size is not None:
            self.batch_size = batch_size
        device = self.get_param_device()
        index_list = list(range(self.dropout_mask_nums))
        index_tile_list = index_list * math.ceil(self.batch_size / self.dropout_mask_nums)
        indexs = torch.tensor(random.sample(index_tile_list, self.batch_size), device=device)
        self.hidden1_mask_select = torch.index_select(self.hidden1_mask, 1, indexs)
        self.hidden2_mask_select = torch.index_select(self.hidden2_mask, 1, indexs)
        self.hidden3_mask_select = torch.index_select(self.hidden3_mask, 1, indexs)

    def get_param_device(self):
        return next(self.parameters()).device

    def compute_decays(self):
        """ 计算权重衰减loss """
        lin0_decays = 0.0001 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.00025 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.00025 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.0005 * (self.lin3_w ** 2).sum() / 2.0
        return lin0_decays + lin1_decays + lin2_decays + lin3_decays


