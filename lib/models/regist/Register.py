"""1.	导入必要的库：
o	torch 和 torch.nn 用于构建神经网络。
o	deepcopy 用于创建对象的深拷贝。
o	logger 用于记录信息。
o	MODEL 用于注册模型模块。
o	param_size 用于计算模型参数的数量。
o	rot6d_to_rotmat, euler_to_quat, quat_to_rot6d, quat_to_rotmat 用于不同旋转表示之间的转换。"""
import torch
import torch.nn as nn
from copy import deepcopy
from lib.utils.logger import logger
from lib.utils.builder import MODEL
from lib.utils.misc import param_size
from lib.utils.transform import rot6d_to_rotmat, euler_to_quat, quat_to_rot6d, quat_to_rotmat

"""注册模型模块：
使用 @MODEL.register_module() 装饰器将 Register 类注册为一个模型模块。"""
@MODEL.register_module()
class Register(nn.Module):

"""初始化函数 __init__：
接受配置 cfg 和初始化数据 init_data。
将欧拉角 INIT_R 转换为四元数 quat，然后再转换为 6D 旋转向量 rot6d。
定义三个可训练的参数：旋转 r，平移 t 和缩放 s。"""
    def __init__(self, cfg, init_data):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        euler = torch.tensor(init_data.INIT_R, dtype=torch.float32) * torch.pi / 180
        quat = euler_to_quat(euler)
        rot6d = quat_to_rot6d(quat)

        self.r = nn.Parameter(rot6d, requires_grad=True)
        self.t = nn.Parameter(torch.tensor(init_data.INIT_T, dtype=torch.float32), requires_grad=True)
        self.s = nn.Parameter(torch.tensor(init_data.INIT_S, dtype=torch.float32), requires_grad=True)

        logger.info(f"{self.name} has {param_size(self)}M parameters")

"""训练设置 training_setup：
根据训练参数 training_args 设置优化器 optimizer，为旋转、平移和缩放设置不同的学习率。"""
    def training_setup(self, training_args):
        l = [{
            'params': [self.r],
            'lr': training_args.R_LR,
            "name": "r"
        }, {
            'params': [self.t],
            'lr': training_args.T_LR,
            "name": "t"
        }, {
            'params': [self.s],
            'lr': training_args.S_LR,
            "name": "s"
        }]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

"""获取缩放属性 get_scale：
返回缩放参数 s 的一个深拷贝，并且从计算图中分离出来。"""
    @property
    def get_scale(self):
        return deepcopy(self.s).detach()

"""前向传播函数 forward：
将 6D 旋转向量 r 转换为 3x3 旋转矩阵 R。
计算输入点云 xyz 的质心 origin，并对点云进行缩放和平移变换。
应用旋转矩阵 R 到变换后的点云上，然后加上平移向量 self.t。"""
    def forward(self, xyz):
        R = rot6d_to_rotmat(self.r)

        origin = torch.mean(xyz, dim=0, keepdim=True)
        xyz = self.s * (xyz - origin)  #+ origin

        xyz = (R @ xyz.transpose(0, 1)).transpose(0, 1) + self.t.unsqueeze(0)

        return xyz
