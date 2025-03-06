import os
import json
import math
import numpy as np
from copy import deepcopy
from typing import NamedTuple
from PIL import Image
from termcolor import colored
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.builder import DATASET

# 定义一个命名元组 CameraInfo，用于存储相机的相关信息
class CameraInfo(NamedTuple):
    uid: int # 相机的唯一标识符
    R: np.array  # 相机的旋转矩阵
    T: np.array  # 相机的平移向量
    FovY: np.array  # 相机的垂直视野角度
    FovX: np.array  # 相机的水平视野角度
    image: np.array  # 相机拍摄的图像
    image_path: str  # 图像文件的路径
    image_name: str  # 图像文件的名称
    width: int  # 图像的宽度
    height: int  # 图像的高度

# 定义一个函数，用于将焦距转换为视野角度
def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

# 使用装饰器将 MPM_Synthetic 类注册到 DATASET 模块中
@DATASET.register_module()
class MPM_Synthetic():

    def __init__(self, cfg) -> None:
        # 获取当前类的名称
        self.name = type(self).__name__
        # 存储配置信息
        self.cfg = cfg

        # 从配置信息中获取数据根目录
        self.data_root = cfg.DATA_ROOT
        # 从配置信息中获取对象的名称
        self.obj_id = cfg.OBJ_NAME
        # 从配置信息中获取相机的数量
        self.num_views = cfg.N_CAM
        # 从配置信息中获取帧数
        self.n_frames = cfg.N_FRAME
        # 从配置信息中获取所有帧数
        self.all_frames = cfg.FRAME_ALL
        # 从配置信息中获取撞击帧
        self.hit_frame = cfg.HIT_FRAME
        # 从配置信息中获取图像的高度
        self.H = cfg.H
        # 从配置信息中获取图像的宽度
        self.W = cfg.W

        # 对象的名称
        self.obj_name = self.obj_id
         # 数据的路径，由数据根目录和对象名称拼接而成
        self.data_path = os.path.join(self.data_root, self.obj_name)

        self.white_bkg = False # 是否使用白色背景，默认为 False
        self.bg = np.array([0, 0, 0])  # 背景颜色，默认为黑色

        # 记录日志，输出对象名称、相机数量、帧数等信息
        logger.info(f"{self.name}: {colored(self.obj_id, 'yellow', attrs=['bold'])} has "
                    f"{colored(self.num_views, 'yellow', attrs=['bold'])} cameras "
                    f"with {colored(self.n_frames, 'yellow', attrs=['bold'])}/"
                    f"{colored(self.all_frames, 'yellow', attrs=['bold'])} frames")

    def get_cam_info(self):
        with open(os.path.join(self.data_path, 'camera.json'), 'r') as cam_file:
            cameras = json.load(cam_file)

        logger.info('Reading all data ...') # 记录日志，提示正在读取所有数据

        cam_infos_all = [] # 用于存储所有帧的相机信息
         # 遍历每一帧
        for frame_id in etqdm(range(self.n_frames)):
            cam_infos = []# 用于存储当前帧的相机信息
            # 遍历每一个相机
            for cam_id, camera in enumerate(cameras):
                 # 获取相机的内参矩阵
                intrinsic = np.array(camera['K'])
                # 深拷贝相机的外参矩阵（相机到世界的变换矩阵）
                c2w = deepcopy(np.array(camera['c2w']))
                 # 对 c2w 矩阵的部分元素取反
                c2w[:3, 1:3] *= -1
                  # 计算世界到相机的变换矩阵
                w2c = np.linalg.inv(c2w)
                # 提取旋转矩阵并进行转置
                R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3] # 提取平移向量

                FovX = focal2fov(intrinsic[0][0], self.W)  # 计算相机的水平视野角度
                FovY = focal2fov(intrinsic[1][1], self.H) # 计算相机的垂直视野角度

                cam_name = camera['camera']
                image_path = os.path.join(self.data_path, cam_name, f"{frame_id:03}.png") # 拼接当前帧图像的文件路径
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA")) # 将图像转换为 RGBA 格式并转换为 numpy 数组
                norm_data = im_data / 255.0    # 对图像数据进行归一化处理
                # 根据透明度通道合成图像，将背景颜色与图像进行融合
                arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
                 # 将合成后的图像数据转换为字节类型，并转换为 RGB 格式的图像
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

                # 创建一个 CameraInfo 对象，并添加到当前帧的相机信息列表中
                cam_infos.append(
                    CameraInfo(uid=cam_id,
                               R=R,
                               T=T,
                               FovY=FovY,
                               FovX=FovX,
                               image=image,
                               image_path=image_path,
                               image_name=f"{cam_name}_{frame_id:03}.png",
                               width=self.W,
                               height=self.H))

            # 将当前帧的相机信息添加到所有帧的相机信息列表中
            cam_infos_all.append(cam_infos)

           # 返回所有帧的相机信息
        return cam_infos_all
