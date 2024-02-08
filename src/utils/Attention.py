# 下面是通道注意力机制即SEnet的代码，个人理解SEnet是在对每个channel进行加权操作，看那个channel重要，就给他权重大一点。
# 每个channel由于每个特征向量不一样因此提取的特征也就不一样即通道注意力机制注重的是特征的重要性（图片的点、线、明暗）
# 因此通道注意力机制是对每个通道的长宽进行卷积，一开始对通道不进行改变
import torch
import torch.nn as nn
import math

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 下面是通道注意力机制和空间注意力机制的结合（CBAM），空间注意力机制是每个通道内的看哪个比较重要
# 空间注意力机制他对应的将通道这个维度进行卷积或者池化，对通道进行压缩操作使其变成两个通道，对图片的长宽不进行改变
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


# 下面是ECA，其就是将通道注意力机制中的两个全连接操纵变成了一个一维卷积
class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 上面相当于对每个不同的图片，卷积核去自适应的进行改变

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        # padding=(kernel_size - 1) // 2 相当于paddin=same，即保持输出图片大小不变的操作
        # 为啥这里进入的通道数是1呢，是因为前面有个自适应层，将图片变成了1*1*channel这个样子，在下面经过维度变换，此时将维度变成了b*1*c，然后conv1d是对最后一维进行卷积的（同理conv2d是对最后两维进行卷积的）因此就是对channel这个维度进行了一个卷积，此时就可以相当于把一个长方体横过来看（或者说换成了channel和长这个面）此时相当于宽为以前的通道数即1.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y.squeeze(-1)是将最后一个维度删掉即宽这个维度就没有了，transpose(-1, -2)是将最后一个和倒数第二个维度进行互换，即现在的维度变成了b，1，c这三个维度，1是由于前面的自适应平均层变成了1*1的图像，所以长在这里就是1。unsqueeze(-1)是增加最后一个维度
        y = self.sigmoid(y)
        return x * y.expand_as(x)
        # y.expand_as(x)是将y的size于x的size进行一个统一，可以看成将y像x一样扩展

class Eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(Eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


