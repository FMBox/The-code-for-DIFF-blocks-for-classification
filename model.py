'''
Define model structure
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from torch import nn, einsum
from torch import einsum
from einops import rearrange



'''1. CNN branch: Local features - ConvBlock'''
class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, res_conv=True, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        med_planes = 64
        conv_1x1 = default_conv_1x1
        conv_3x3 = default_conv_3x3


        # 1x1 conv: shrink channel dimensions 
        self.conv1 = conv_1x1(in_channels=med_planes, out_channels=med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        # 3x3 conv: extract local spatial features 
        self.conv2 = conv_3x3(in_channels=med_planes, out_channels=med_planes, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        # 1x1 conv: restore channels 
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





'''2. Transformer Branch: global'''
# A block: MHSA & MLP

# MHSA
# Compute attention & combine
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




# MLP 
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

# Layer Normalization to embeddings before MHSA/FFN: improve training stability
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




'''3. DIFF block - Feature Fusion units'''
class FB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=False, act=nn.ReLU(True), norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):

        super(FB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            modules_body.append(norm_layer)

            if i == 0:
                modules_body.append(act)
         
        self.body = nn.Sequential(*modules_body)   # 2 conv layers + BatchNorm

    def forward(self, x):
        res = self.body(x)
        res += x
        return res




'''4. My model: CNN-Transformer'''
class ConTrans(nn.Module):
    def __init__(self, num_classes=2):

        super(ConTrans, self).__init__()

        conv = default_conv

        task = 'sr'
        n_feats = 64
        n_resgroups = 4          
        n_resblocks = 1          
        reduction = 16
        n_heads = 6             
        n_layers = 8            
        dropout_rate = 0
        n_fusionblocks = 4         
        token_size = 3
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


        '''Stem module'''
        self.out_channels = out_channels = min(n_feats, int(n_feats * 2))
        self.head = nn.Sequential(
            nn.Conv2d(n_colors, 64, kernel_size=7, stride=1, padding=3, bias=False),       
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),      
        )


        '''Parallel CNN & Trans'''
        
        '''CNN Branch'''
        self.cnn_branch = nn.ModuleList([
            nn.Sequential(
                ConvBlock(inplanes=n_feats, outplanes=n_feats, res_conv=True, stride=1),
            ) for _ in range(n_resgroups)
        ])



        '''Transformer Branch''' 
        # Trans Block: MHSA & MLP 
        self.linear_encoding = nn.Linear(flatten_dim, embedding_dim)

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

        
        '''DIFF: Feature Fusion Block'''
        # each DIFF: 2 FB
        self.fusion_block = nn.ModuleList([
            nn.Sequential(
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True), norm_layer=nn.BatchNorm2d(n_feats * 2)),
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True), norm_layer=nn.BatchNorm2d(n_feats * 2)),
            ) for _ in range(n_fusionblocks)
        ])

        
        self.conv_temp = conv(n_feats * 2, n_feats, 3)
        
        # single convolutional layer: lessen dimension 
        self.conv_last = conv(n_feats * 2, n_feats, 3)      


        '''classification layer'''
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))      
        self.fc = nn.Linear(n_feats, num_classes)         


    def forward(self, x):
        h, w = x.shape[-2:]
        
        '''Stem module'''
        x = self.head(x)
        
        identity = x

        '''Parallel CNN & Trans'''
        
        '''Trans Branch: embeddings -> MHSA, MLP -> ...'''
        # Convert feat into embed
        x_tkn = F.unfold(x, self.token_size, stride=self.token_size)     
        x_tkn = rearrange(x_tkn, 'b d t -> b t d')     
        x_tkn = self.linear_encoding(x_tkn) + x_tkn

        # MHSA & MLP
        for i in range(self.n_fusionblocks):
            x_tkn = self.mhsa_block[i][0](x_tkn) + x_tkn            # Transformer features
            x_tkn = self.mhsa_block[i][1](x_tkn) + x_tkn

            x = self.cnn_branch[i](x)           # CNN features 

            x_tkn_res, x_res = x_tkn, x   

            x_tkn = rearrange(x_tkn, 'b t d -> b d t')
            # Convert embed into feat
            x_tkn = F.fold(x_tkn, (h, w), self.token_size, stride=self.token_size)   


            '''DIFF Block'''
            f = torch.cat((x, x_tkn), 1)        # Concat 
            
            f = f + self.fusion_block[i](f)        # DIFF units
            
            # a conv -> feed to CNN & Trans
            if i != (self.n_fusionblocks - 1):
                x_temp = self.conv_temp(f)
                x = x_temp                 # CNN 
                x_tkn = x_temp             # Trans 

                x_tkn = F.unfold(x_tkn, self.token_size, stride=self.token_size)
                x_tkn = rearrange(x_tkn, 'b d t -> b t d')
            
            
            
        # final fusion
        x = self.conv_last(f)     # a conv
        x = x + identity

        
        '''Classification layer: FC'''
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



# Create CNN-Trans model with num_classes
def create_CNNTrans0(deploy=False):
    return ConTrans(num_classes=6)                   
func_dict = {
'CNNTrans0': create_CNNTrans0,
}
def get_CNNTrans_func_by_name(name):
    return func_dict[name]

