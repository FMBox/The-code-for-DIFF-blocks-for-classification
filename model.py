import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
import math
from einops import rearrange


## CNN branch
class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()
     
        med_planes = 64

        conv_1x1 = default_conv_1x1
        conv_3x3 = default_conv_3x3

        self.conv1 = conv_1x1(in_channels=med_planes, out_channels=med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = conv_3x3(in_channels=med_planes, out_channels=med_planes, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = conv_1x1(in_channels=med_planes, out_channels=med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = conv_1x1(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        return x



## Fusion Blocks
class FB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=False, act=nn.ReLU(True), norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):

        super(FB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            modules_body.append(norm_layer)

            if i == 0:
                modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


def default_conv_1x1(in_channels, out_channels, kernel_size, stride, padding, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding, bias=bias
    )

def default_conv_3x3(in_channels, out_channels, kernel_size, stride, padding, groups, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding, groups, bias=bias
    )

class Upsampler(nn.Sequential):
    def __init__(self,
                 conv,
                 scale,
                 n_feats,
                 bn=False,
                 act=False,
                 bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm2(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=6, dim_head=64, dropout=0.0):

        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)



## My model
class ConTrans(nn.Module):
    def __init__(self, num_classes=2):

        super(ConTrans, self).__init__()

        conv = default_conv
     
        task = 'sr'
        n_feats = 64
        n_resgroups = 4          
        n_resblocks = 1          
        reduction = 16
        n_heads = 8             
        n_layers = 8            
        dropout_rate = 0
        n_fusionblocks = 4         
        token_size = 4
        expansion_ratio = 4
        scale = 1
        rgb_range = 255
        n_colors = 3

        self.num_classes = num_classes
        self.n_feats = n_feats
        self.token_size = token_size
        self.n_fusionblocks = n_fusionblocks
        self.embedding_dim = embedding_dim = n_feats * (token_size ** 2)     

        flatten_dim = embedding_dim
        hidden_dim = embedding_dim * expansion_ratio
        dim_head = embedding_dim // n_heads

        # stem module 
        self.out_channels = out_channels = min(n_feats, int(n_feats * 2))
        self.head = nn.Sequential(
            nn.Conv2d(n_colors, 64, kernel_size=7, stride=1, padding=3, bias=False),       
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),      
        )

        self.linear_encoding = nn.Linear(flatten_dim, embedding_dim)

        # attention
        self.mhsa_block = nn.ModuleList([
            nn.ModuleList([
                PreNorm(
                    embedding_dim,
                    SelfAttention(embedding_dim, n_heads, dim_head, dropout_rate)
                ),
                PreNorm(
                    embedding_dim,
                    FeedForward(embedding_dim, hidden_dim, dropout_rate)
                ),
            ]) for _ in range(n_layers // 2)
        ])

        # CNN Branch
        self.cnn_branch = nn.ModuleList([
            nn.Sequential(
                ConvBlock(inplanes=n_feats, outplanes=n_feats, res_conv=True, stride=1),
            ) for _ in range(n_resgroups)
        ])
        
        # DIFF Blocks
        self.fusion_block = nn.ModuleList([
            nn.Sequential(
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True), norm_layer=nn.BatchNorm2d(n_feats * 2)),
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True), norm_layer=nn.BatchNorm2d(n_feats * 2)),
            ) for _ in range(n_fusionblocks)
        ])

        self.conv_temp = conv(n_feats * 2, n_feats, 3)       
    
        # single convolution
        self.conv_last = conv(n_feats * 2, n_feats, 3)              

        # classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))      
        self.fc = nn.Linear(n_feats, num_classes)         


    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.head(x)
       
        identity = x

        # feature map to patch embedding
        x_tkn = F.unfold(x, self.token_size, stride=self.token_size)     
        x_tkn = rearrange(x_tkn, 'b d t -> b t d')     

        # linear encoding after tokenization
        x_tkn = self.linear_encoding(x_tkn) + x_tkn

        for i in range(self.n_fusionblocks):
            x_tkn = self.mhsa_block[i][0](x_tkn) + x_tkn       # MHSA in Transformer block
            x_tkn = self.mhsa_block[i][1](x_tkn) + x_tkn
         
            x = self.cnn_branch[i](x)     # CNN Branch 

            x_tkn_res, x_res = x_tkn, x

            x_tkn = rearrange(x_tkn, 'b t d -> b d t')
            # patch embedding to feature map
            x_tkn = F.fold(x_tkn, (64, 64), self.token_size, stride=self.token_size)   

            f = torch.cat((x, x_tkn), 1)        # Concat CNN and Transformer
            f = f + self.fusion_block[i](f)        # DIFF Blocks

            if i != (self.n_fusionblocks - 1):
                x_temp = self.conv_temp(f)
                x = x_temp
                x_tkn = x_temp  
                x_tkn = F.unfold(x_tkn, self.token_size, stride=self.token_size)
                x_tkn = rearrange(x_tkn, 'b d t -> b t d')
                        
        x = self.conv_last(f)     # single convolution

        x = x + identity
       
        # classification layer
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def create_ConTrans(deploy=False):
    return ConTrans(num_classes=5)                   
func_dict = {
'ConTrans': create_ConTrans,
}
def get_ConTrans_func_by_name(name):
    return func_dict[name]

