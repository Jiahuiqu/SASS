import math

import numpy as np
import torch.nn.functional as F

from einops import rearrange, repeat
import torch
import torch.nn as nn
import dgl.function as fn
import dgl
import utils


# 梯度反转层
# class grl_func(torch.autograd.Function):
#     def __init__(self):
#         super(grl_func, self).__init__()
#
#     @staticmethod
#     def forward(ctx, x, lambda_):
#         ctx.save_for_backward(lambda_)
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         lambda_, = ctx.saved_variables
#         grad_input = grad_output.clone()
#         return - lambda_ * grad_input, None
#
#
# class GRL(nn.Module):
#     def __init__(self):
#         super(GRL, self).__init__()
#         # self.lambda_ = torch.tensor(lambda_)
#
#     def set_lambda(self, lambda_):
#         self.lambda_ = torch.tensor(lambda_)
#
#     def forward(self, x, lambda_):
#         lambda_ = torch.tensor(lambda_)
#         return grl_func.apply(x, lambda_)


# class MLPReadout(nn.Module):
#
#     def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
#         super().__init__()
#         list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
#         list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
#         self.FC_layers = nn.ModuleList(list_FC_layers)
#         self.L = L
#
#     def forward(self, x):
#         y = x
#         for l in range(self.L):
#             y = self.FC_layers[l](y)
#             y = F.relu(y)
#         y = self.FC_layers[self.L](y)
#         return y

# cnn
# 对原始数据做预处理
class Cascade(nn.Module):
    def __init__(self, ch_in, ch_out, batch):
        super(Cascade, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(ch_in, ch_in * 2, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(ch_in * 2, ch_in, kernel_size=1, stride=1))
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(ch_in, ch_in * 2, kernel_size=3, stride=1, padding=1)
        if batch:
            self.bn = nn.BatchNorm2d(ch_in * 2)
        else:
            self.bn = nn.Identity()
        self.conv3 = nn.Conv2d(ch_in * 2, ch_out, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.LeakyReLU()
        self.skip = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.act1(x1) + x
        x3 = self.bn(self.conv2(x2))
        x3 = self.conv3(x3) + self.skip(x1)
        out = self.act2(x3)
        return out


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out, batch: bool):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_mid, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        if batch:
            self.bn1 = nn.BatchNorm2d(ch_mid)
            self.bn2 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(in_channels=ch_mid, out_channels=ch_out, kernel_size=3, stride=1, padding=1)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1))
            # nn.BatchNorm2d(ch_out))

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = out + self.extra(x)
        return out


class CNN(nn.Module):
    def __init__(self, bands: int, FM: int, batch: bool):
        super(CNN, self).__init__()
        self.conv_HSI_1 = ResBlk(bands, FM * 2, FM, batch=batch)
        self.conv_LIDAR_1 = nn.Sequential(nn.Conv2d(1, FM, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(FM),
                                          nn.LeakyReLU())
        self.conv_HSI_2 = ResBlk(FM, FM * 4, FM * 2, batch=batch)
        self.conv_LIDAR_2 = Cascade(FM, FM * 2, batch)
        self.pool_LIDAR = nn.AvgPool2d(2)
        self.pool_HSI = nn.AvgPool2d(2)
        # self.conv_HSI_3 = ResBlk(FM * 2, FM * 4, FM * 4, batch=batch)
        # self.conv_LIDAR_3 = Cascade(FM * 2, FM * 4, batch)

    def forward(self, data_HSI, data_LiDAR):
        data_HSI = self.conv_HSI_1(data_HSI)
        data_HSI = self.conv_HSI_2(data_HSI)
        data_LiDAR = self.conv_LIDAR_1(data_LiDAR)
        data_LiDAR = self.conv_LIDAR_2(data_LiDAR)
        data_HSI = self.pool_HSI(data_HSI)
        # data_HSI= F.leaky_relu(data_HSI)
        data_LiDAR = F.leaky_relu(self.pool_LIDAR(data_LiDAR))
        # data_HSI = self.conv_HSI_3(data_HSI)
        # data_LiDAR = self.conv_LIDAR_3(data_LiDAR)

        return data_HSI, data_LiDAR


# GTN
# ###############################################
# 构建主动学习
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        # Compute attention score
        # 通过src节点k和dst节点的q的乘积计算注意力分数
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)
        # 使用规模归一化常数对指数进行指数化
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        src_mul_edge = fn.message._gen_message_builtin('u', 'e', 'mul')
        dgl_sum = fn.reducer._gen_reduce_builtin('sum')
        g.send_and_recv(eids, src_mul_edge('V_h', 'score', 'V_h'), dgl_sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), dgl_sum('score', 'z'))

    def forward(self, g, h):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        head_out = g.ndata['wV'] / g.ndata['z']
        index = torch.isnan(head_out)
        head_out[index] = 0

        return head_out


class GraphTransformerLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, use_bias=True, residual=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)

        self.O = nn.Linear(out_dim, out_dim)
        self.residual = residual

        self.norm1 = nn.BatchNorm1d(out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        self.norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):

        h_in1 = h  # for first residual connection

        # multi-head attention out
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.out_channels)

        # h = self.O(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        h = self.norm1(h)

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = self.FFN_layer2(h)
        if self.residual:
            h = h_in2 + h  # residual connection

        h = self.norm2(h)

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}'.format(self.__class__.__name__,
                                                                     self.in_channels,
                                                                     self.out_channels, self.num_heads)


class GraphTransformerNet(nn.Module):

    def __init__(self, in_dim_node, hidden_dim, out_dim, num_heads):
        super().__init__()

        # self.n_classes = n_classes

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)  # node feat is an integer

        self.layers = nn.ModuleList([])
        self.layers.append(GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, True, residual=False))
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, True, residual=False))

    def forward(self, g, h):
        # input embedding

        h = self.embedding_h(h)

        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, h)
            del g.ndata['z']
            del g.ndata['wV']

        # # output
        # h_out = self.MLP_layer(h)

        return h


# ViT
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel=1):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        # self.skipcat = nn.ModuleList([])
        # for _ in range(depth - 2):
        #     self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):

        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class SASS(torch.nn.Module):
    def __init__(self, bands, FM: int, Classes: int, patchsize, batch: bool):
        super(SASS, self).__init__()

        self.patchsize = patchsize
        self.com = CNN(bands, FM, batch)
        self.pri_lbl = CNN(bands, FM, batch)
        self.pri_ulb = CNN(bands, FM, batch)
        self.recon_HSI = nn.Sequential(
            nn.Upsample(scale_factor=(patchsize - 4) / (patchsize // 2)),
            nn.ConvTranspose2d(FM * 2, FM * 4, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(FM * 4, bands, kernel_size=3, stride=1),
            nn.Sigmoid()
        )
        self.recon_LiDAR = nn.Sequential(
            nn.Upsample(scale_factor=(patchsize - 4) / (patchsize // 2)),
            nn.ConvTranspose2d(FM * 2, FM, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(FM, 1, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

        self.cls_token_com = nn.Parameter(torch.randn(1, 1, FM * 4))
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, (patchsize // 2) ** 2 + 1, FM * 4))
        self.pool = nn.AdaptiveAvgPool2d(1)

        # vit
        self.ViT = Transformer(dim=FM * 4, depth=5, heads=4, dim_head=16, mlp_head=8, dropout=0.3)

        self.cls = nn.Sequential(
            nn.LayerNorm(FM * 4),
            nn.Linear(FM * 4, Classes)
        )

    def forward(self, data_HSI, data_LiDAR, method='label', option='train'):

        # 数据预处理
        if option == 'train':
            batch = data_HSI.shape[0]
            data_HSI_com, data_LiDAR_com = self.com(data_HSI, data_LiDAR)
            data_fusion = torch.cat([data_HSI_com, data_LiDAR_com], dim=1)
            feature_local = self.pool(data_fusion)
            feature_local = feature_local.flatten(1)

            # [b,c,h,w]->[b,c,h*w]
            data_fusion = data_fusion.flatten(2)
            # [b,c,l]->[n,l,d]得到vit的输入
            data_fusion = torch.einsum('ndl->nld', data_fusion)
            # 加位置编码和clstoken
            data_fusion = data_fusion + self.encoder_pos_embed[:, 1:, :]
            cls_token = repeat(self.cls_token_com, '() l d->b l d', b=batch)
            data_fusion = torch.cat([cls_token, data_fusion], dim=1)
            data_fusion += self.encoder_pos_embed[:, 0]
            data_fusion = self.ViT(data_fusion)

            # 提取域不变特征和域特有特征
            if method == 'label':
                data_HSI_pri, data_LiDAR_pri = self.pri_lbl(data_HSI, data_LiDAR)

            elif method == 'unlabel':
                data_HSI_pri, data_LiDAR_pri = self.pri_ulb(data_HSI, data_LiDAR)

            # 重建输入数据
            # data_HSI_recon = torch.cat([data_HSI_com, data_HSI_pri], dim=1)
            data_HSI_recon = data_HSI_com + data_HSI_pri
            data_HSI_recon = self.recon_HSI(data_HSI_recon)
            loss_recon_HSI = utils.re_loss(data_HSI, data_HSI_recon)

            # data_LiDAR_recon = torch.cat([data_LiDAR_com, data_LiDAR_pri], dim=1)
            data_LiDAR_recon = data_LiDAR_com + data_LiDAR_pri
            data_LiDAR_recon = self.recon_LiDAR(data_LiDAR_recon)
            loss_recon_LiADR = utils.re_loss(data_LiDAR, data_LiDAR_recon)
            loss_recon = loss_recon_HSI + loss_recon_LiADR

            # 正交
            loss_oc_HSI = utils.diff_loss(data_HSI_com, data_HSI_pri)
            loss_oc_LiDAR = utils.diff_loss(data_LiDAR_com, data_LiDAR_pri)
            loss_oc = loss_oc_HSI + loss_oc_LiDAR

            # 分类
            feature_global = data_fusion[:, 0]
            feature_global = feature_global.view(batch, -1)

            feature = feature_local + feature_global
            # feature=torch.cat([feature_global, feature_local], dim=1)
            result = self.cls(feature)

        elif option == 'test':
            batch = data_HSI.shape[0]
            data_HSI_com, data_LiDAR_com = self.com(data_HSI, data_LiDAR)
            data_fusion = torch.cat([data_HSI_com, data_LiDAR_com], dim=1)
            feature_local = self.pool(data_fusion)
            feature_local = feature_local.flatten(1)

            # [b,c,h,w]->[b,c,h*w]
            data_fusion = data_fusion.flatten(2)
            # [b,c,l]->[n,l,d]得到vit的输入
            data_fusion = torch.einsum('ndl->nld', data_fusion)
            # 加位置编码和clstoken
            data_fusion = data_fusion + self.encoder_pos_embed[:, 1:, :]
            cls_token = repeat(self.cls_token_com, '() l d->b l d', b=batch)
            data_fusion = torch.cat([cls_token, data_fusion], dim=1)
            data_fusion += self.encoder_pos_embed[:, 0]
            data_fusion = self.ViT(data_fusion)

            # 分类
            feature_global = data_fusion[:, 0]
            feature_global = feature_global.view(batch, -1)

            feature = feature_local + feature_global
            # feature=torch.cat([feature_global, feature_local], dim=1)
            result = self.cls(feature)

            loss_recon = 0
            loss_oc = 0

        return result, feature, {'global': feature_global, 'hsi': data_HSI_com, 'lidar': data_LiDAR_com,
                                 'recon': loss_recon, 'diff': loss_oc}


class dis(nn.Module):
    def __init__(self, FM):
        super(dis, self).__init__()

        self.out = GraphTransformerNet(FM * 4, FM * 2, FM, 4)
        self.out2 = nn.Sequential(
            nn.Linear(FM, 1),
            nn.Sigmoid()
        )

    def forward(self, g, h):
        h = self.out(g, h)
        y = self.out2(h)
        y = torch.clamp(y, 1e-5, 1 - 1e-5)
        # y = self.out3(h)
        return y

from torch_geometric.nn import GCNConv

