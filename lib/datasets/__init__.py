python
from lib.utils.config import CN  # 导入配置节点类(Config Node)，用于处理层级化配置

from ..utils.builder import build_dataset  # 导入数据集构建器函数
from .mpm_synthetic import MPM_Synthetic  # 导入合成数据集类
from .real_capture import Real_Capture  # 导入真实捕获数据集类

def create_dataset(cfg: CN, data_preset: CN):
    """
    创建数据集实例（带数据预设配置）
    
    参数:
        cfg (CN): 主配置对象，包含全局配置参数
        data_preset (CN): 数据预设配置，包含数据集特定的初始化参数
    
    返回:
        Dataset: 实例化的数据集对象
    
    功能:
        1. 结合主配置和数据预设配置
        2. 使用构建器模式实例化对应数据集
        3. 主要用于训练/验证阶段需要数据预设的场景
    """
    return build_dataset(cfg, data_preset=data_preset)  # 调用构建器创建带预设配置的数据集

def load_dataset(cfg: CN):
    """
    加载数据集实例（仅需主配置）
    
    参数:
        cfg (CN): 主配置对象，包含数据集类型和基本参数
    
    返回:
        Dataset: 实例化的数据集对象
    
    功能:
        1. 直接从主配置解析数据集参数
        2. 使用构建器模式实例化数据集
        3. 主要用于推理/测试阶段无需额外预设的场景
    """
    return build_dataset(cfg)  # 调用构建器创建基础配置的数据集
