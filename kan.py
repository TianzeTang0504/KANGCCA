import torch
from kannet.utils import *
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from kannet import kannet


class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=10
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        if not no_kan:
            self.fc1 = kannet.KANLinear(
                        in_features,
                        hidden_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc2 = kannet.KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc3 = kannet.KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)
    
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
    
        return x


class KANBlock(nn.Module):
    def __init__(self, dim, mlp_hidden_dim, out_dim, drop=0.1, drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.drop_path(self.layer(self.norm2(x)))

        return x



class UKAN(nn.Module):
    def __init__(self, layer_sizes, input_size, embed_dims=[128, 160, 256], drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm, depths=[1, 1, 1]):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.fc_in = nn.Linear(input_size, 128)

        self.block1 = nn.ModuleList([KANBlock(
            dim = embed_dims[0],
            mlp_hidden_dim = embed_dims[1],
            out_dim = embed_dims[1],
            drop = drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.block2 = nn.ModuleList([KANBlock(
            dim = embed_dims[1],
            mlp_hidden_dim = embed_dims[2],
            out_dim = embed_dims[2],
            drop = drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim = embed_dims[2],
            mlp_hidden_dim = embed_dims[1],
            out_dim = embed_dims[1],
            drop = drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim = embed_dims[1],
            mlp_hidden_dim = embed_dims[0],
            out_dim = embed_dims[0],
            drop = drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.fc_out = nn.Linear(128, 128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

    def forward(self, x):

        out = self.fc_in(x)

        for i, blk in enumerate(self.block1):
            out = blk(out)
        out = self.norm3(out)

        for i, blk in enumerate(self.block2):
            out = blk(out)
        out = self.norm4(out)

        for i, blk in enumerate(self.dblock1):
            out = blk(out)
        out = self.dnorm3(out)

        for i, blk in enumerate(self.dblock2):
            out = blk(out)
        out = self.dnorm4(out)

        out = self.fc_out(out)

        return out
