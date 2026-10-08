# 针对多块石头爆堆图片密集的特性改进的位置编码 (即插即用)
# 使用深度可分离卷积对标准的正弦位置编码进行局部特征增强，以改善在密集、多实例情况下的位置感知能力，特别是边界特征。

import math
import torch
from torch import nn
from maskdino.modeling.pixel_decoder.position_encoding import PositionEmbeddingSine

class EnhancedPositionEmbeddingSine(nn.Module):
    """
    改进的位置编码:
    在原始正弦位置编码(Sine PE)的基础上，加入一个轻量级的局部卷积增强模块 (类似于CPE / PEG 的思想)。
    这样不仅可以保留全局的绝对位置信息，还可以通过卷积融合局部的结构信息，在面对一张图片中有多块形状各异、
    排列密集的石头爆堆图片时，可以更好地增强位置特征的区分度。
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.sine_pe = PositionEmbeddingSine(num_pos_feats, temperature, normalize, scale)
        
        # 正弦位置编码的输出通道数通常为 2 * num_pos_feats (pos_x 和 pos_y 各占 num_pos_feats)
        dim = num_pos_feats * 2
        
        # 增加一个局部特征增强模块，使用深度可分离卷积以降低参数量
        self.local_enhancer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
        )

    def forward(self, x, mask=None):
        # 1. 计算原始的 Sine 位置编码
        pos = self.sine_pe(x, mask)
        
        # 2. 对位置编码进行局部上下文增强
        enhanced_pos = pos + self.local_enhancer(pos)
        
        return enhanced_pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.sine_pe.num_pos_feats),
            "temperature: {}".format(self.sine_pe.temperature),
            "normalize: {}".format(self.sine_pe.normalize),
            "scale: {}".format(self.sine_pe.scale),
            "local_enhancer: added",
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
