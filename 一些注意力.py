
#注意力的作用，抑制不相关（通道，像素），对（通道，像素等）进行重新校准
from torch import nn
import torch
class senet(nn.Module):



    def __init__(self,inchannels,ratio = 16):
        super(senet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Sequential(
            # 第一个全连接层：
            # 这一层实现了降维（squeeze）的操作。通过将输入的全局特征向量映射到较小的维度，模型可以学习全局信息中的抽象表示。
            # 使用ReLU激活函数引入非线性，允许模型学习更复杂的特征

            nn.Linear(inchannels,inchannels//16,False),
            nn.ReLU(),
            # 第二个全连接层：
            # 这一层实现了升维（excitation）的操作。通过将降维后的特征映射回原始通道数，模型可以为每个通道学习一个权重，表示该通道相对于整体任务的重要性。
            # 使用Sigmoid激活函数将输出值压缩到 0 到 1 的范围内，形成一个通道的注意力权重


            nn.Linear(inchannels//16,inchannels,False),
            nn.Sigmoid(),
        )
        #第一个nn.Linear层减少通道数，这样做的目的是减少参数数量，降低过拟合风险，并减少计算量。这一步可以看作是特征压缩，它将高维的特征空间映射到一个低维的空间中，以便捕获通道间的全局依赖性。
        #nn.ReLU层用于引入非线性，它使网络能够学习更复杂的特征表示。
        #第二个nn.Linear层将通道数增加回原来的数量，这样做是为了恢复特征的维度，使得每个通道都可以有一个独特的权重，这些权重代表了通道的重要性
        #     与不改变通道数的连续三个全连接层相比，减少通道数的好处：
        # 计算效率：如果保持通道数不变，连续三个全连接层会带来巨大的参数量，这不仅增加了存储和计算成本，还可能导致训练过程中的效率下降。相比之下，通过减少通道数，SE模块显著减少了参数数量，这使得模型更加轻量级和计算效率更高。
        # 避免信息冗余：在高维空间中，特征之间可能存在大量冗余信息。通过减少通道数，我们可以强制网络提炼出更有用的信息，从而提高数据的表示效率。这个过程类似于信息压缩，可以帮助网络学习到更加本质和高效的特征表示。
        # 增加模型的泛化能力：如前所述，减少参数数量可以提高模型的泛化能力。与连续使用三个全连接层相比，通过引入瓶颈层（即减少通道数的层），SE模块能够更有效地控制模型的复杂度，从而提高其在未知数据上的表现。
        # 总结来说，通过在SE模块中先减少后增加通道数，我们不仅减少了过拟合的风险，还提高了计算效率和模型的泛化能力。这种设计使得SE模块在增强网络的同时保持了轻量级和高效性。


    def forward(self,x):
        b,c,h,w = x.size()
        avg = self.avgpool(x).view([b,c])
        fc = self.fc(avg).view([b,c,1,1])

        return x*fc
#cbam
class spatial_attention(nn.Module):
    def __init__(self,kernel_size = 7):
        super(spatial_attention,self).__init__()
        padding =kernel_size//2
        self.conv = nn.Conv2d(2,1,kernel_size,1,padding=padding,bias=False)
        self.sigmoid =nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()
        max_pool = torch.max(x,dim=1,keepdim=True).indices
        mean_pool = torch.mean(x,dim=1,keepdim=True)
        pool_out = torch.cat([max_pool,mean_pool],dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out*x

class cbam_SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(cbam_SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat_out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(cat_out)
        return self.sigmoid(out)
class cbam_ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(cbam_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = cbam_ChannelAttention(in_channels)
        self.spatial_attention = cbam_SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
#坐标注意力
class coord_attention(nn.Module):
    def __init__(self,channel,reduction=16):
        super(coord_attention, self).__init__()
        self._1x1conv = nn.Conv2d(channel,channel//16,kernel_size=1,stride=1,bias=False)
        self.bn       = nn.BatchNorm2d(channel//reduction)
        self.relu     = nn.ReLU()
        self.F_h      = nn.Conv2d(channel//16,channel,kernel_size=1,stride=1,bias=False)
        self.F_w     = nn.Conv2d(channel//16,channel,kernel_size=1,stride=1,bias=False)
    def forward(self,x):
        #b,c,h,w
        _,_,h,w = x.size()

        #b,c,h,w->b,c,h,1->b,c,1,h
        x_h = torch.mean(x,dim=3,keepdim=True).permute(0,1,3,2)

        #b,c,h,w->b,c,1,w
        x_w = torch.mean(x,dim=2,keepdim=True)
        x_cat_bn_relu = self.relu(self.bn(self._1x1conv(torch.cat((x_h,x_w),3))))

        x_cat_split_h,x_cat_split_w = x_cat_bn_relu.split([h,w],3)

        s_h =  torch.sigmoid(self.F_h(x_cat_split_h.permute(0,1,3,2)))
        s_w =  torch.sigmoid(self.F_w(x_cat_split_w))

        out = x*s_h.expand_as(x)*s_w.expand_as(x)
        return out

#交叉注意力：
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x