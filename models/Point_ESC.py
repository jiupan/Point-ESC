import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from utils.resa_utils import ReSALoss, ProjectionHead
from knn_cuda import KNN
from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        Args:
            point_groups: B G N 3
        Returns:
            feature_global: B G C
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, indices=None, return_indices=False):
        """
        Args:
            xyz: B N 3
        Returns:
            neighborhood: B G M 3
            center: B G 3
        """
        batch_size, num_points, _ = xyz.shape
        if indices is None:
            center_idx = pointnet2_utils.furthest_point_sample(xyz, self.num_group).long()
            batch_idx = torch.arange(batch_size, device=xyz.device).unsqueeze(1)
            center = xyz[batch_idx, center_idx]
            _, idx = self.knn(xyz, center)
        else:
            center_idx, idx = indices
            batch_idx = torch.arange(batch_size, device=xyz.device).unsqueeze(1)
            center = xyz[batch_idx, center_idx]
        group_indices = (center_idx, idx)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        if return_indices:
            return neighborhood, center, group_indices
        return neighborhood, center


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        depth=4,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        x = self.head(self.norm(x[:, -return_token_num:]))
        return x


class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f"[args] {config.transformer_config}", logger="Transformer")
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.mask_type = config.transformer_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2], dtype=torch.bool, device=center.device)
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0)
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1)
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        bool_masked_pos = torch.stack(mask_idx).to(center.device)
        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2], dtype=torch.bool, device=center.device)
        num_mask = int(self.mask_ratio * G)
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([np.zeros(G - num_mask), np.ones(num_mask)])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        return overall_mask.to(center.device)

    def forward(self, neighborhood, center, noaug=False):
        if self.mask_type == "rand":
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)
        group_input_tokens = self.encoder(neighborhood)
        batch_size, seq_len, C = group_input_tokens.size()
        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)
        return x_vis, bool_masked_pos


class ImprovedMaskTransformer(MaskTransformer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.consistent_mask = getattr(config.transformer_config, "consistent_mask", True)

    def forward(self, neighborhood, center, noaug=False, shared_mask=None):
        if shared_mask is not None:
            bool_masked_pos = shared_mask
        else:
            if self.mask_type == "rand":
                bool_masked_pos = self._mask_center_rand(center, noaug=noaug)
            else:
                bool_masked_pos = self._mask_center_block(center, noaug=noaug)
        group_input_tokens = self.encoder(neighborhood)
        batch_size, seq_len, C = group_input_tokens.size()
        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)
        return x_vis, bool_masked_pos


@MODELS.register_module()
class Point_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log("[Point_MAE] ", logger="Point_MAE")
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        print_log(
            f"[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...",
            logger="Point_MAE",
        )
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.increase_dim = nn.Sequential(nn.Conv1d(self.trans_dim, 3 * self.group_size, 1))
        trunc_normal_(self.mask_token, std=0.02)
        self.loss = config.loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == "cdl2":
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)
        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B, _, C = x_vis.shape
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        x_rec = self.MAE_decoder(x_full, pos_full, N)
        B, M, C = x_rec.shape
        rebuild_points = (
            self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        )
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)
        if vis:
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            return ret1, ret2, full_center
        return loss1


@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim),
        )
        self.build_loss_func()
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt["base_model"].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith("MAE_encoder"):
                    base_ckpt[k[len("MAE_encoder.") :]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith("base_model"):
                    base_ckpt[k[len("base_model.") :]] = base_ckpt[k]
                    del base_ckpt[k]
            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if incompatible.missing_keys:
                print_log("missing_keys", logger="Transformer")
                print_log(get_missing_parameters_message(incompatible.missing_keys), logger="Transformer")
            if incompatible.unexpected_keys:
                print_log("unexpected_keys", logger="Transformer")
                print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger="Transformer")
            print_log(f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}", logger="Transformer")
        else:
            print_log("Training from scratch!!!", logger="Transformer")
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret


@MODELS.register_module()
class Point_ESC(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log("[Point_ESC] Initializing...", logger="Point_ESC")
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate

        self.MAE_encoder = ImprovedMaskTransformer(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        print_log(
            f"[Point_ESC] divide point cloud into G{self.num_group} x S{self.group_size} points ...",
            logger="Point_ESC",
        )
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.increase_dim = nn.Sequential(nn.Conv1d(self.trans_dim, 3 * self.group_size, 1))

        self.use_resa = getattr(config, "use_resa", True)
        if self.use_resa:
            resa_config = getattr(config, "resa_config", {})
            self.resa_proj_dim = resa_config.get("proj_dim", 256)
            self.resa_hidden_dim = resa_config.get("hidden_dim", 512)
            self.resa_temperature = resa_config.get("temperature", 0.1)
            self.resa_sinkhorn_iters = resa_config.get("sinkhorn_iters", 3)
            self.resa_sinkhorn_epsilon = resa_config.get("sinkhorn_epsilon", 0.05)
            self.resa_similarity_type = resa_config.get("similarity_type", "cosine")
            self.resa_distance_type = resa_config.get("distance_type", "kl")
            self.resa_lambda = resa_config.get("lambda", 0.1)
            self.progressive_training = resa_config.get("progressive_training", True)
            self.resa_start_epoch = resa_config.get("start_epoch", 60)
            self.feature_detach = resa_config.get("feature_detach", False)
            self.projection_head = ProjectionHead(
                input_dim=self.trans_dim,
                hidden_dim=self.resa_hidden_dim,
                output_dim=self.resa_proj_dim,
                num_layers=2,
            )
            self.resa_loss_fn = ReSALoss(
                num_clusters=self.num_group,
                similarity_type=self.resa_similarity_type,
                temperature=self.resa_temperature,
                sinkhorn_iters=self.resa_sinkhorn_iters,
                sinkhorn_epsilon=self.resa_sinkhorn_epsilon,
                distance_type=self.resa_distance_type,
            )
            print_log(
                f"[Point_ESC] ReSA enabled: lambda={self.resa_lambda}, start_epoch={self.resa_start_epoch}",
                logger="Point_ESC",
            )

        trunc_normal_(self.mask_token, std=0.02)
        self.loss = config.loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == "cdl2":
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

    def forward(self, pts, vis=False, epoch=0, view_transform=None, **kwargs):
        if self.use_resa and self.training and view_transform is not None:
            return self._forward_with_resa(pts, view_transform, vis, epoch)
        return self._forward_mae_only(pts, vis)

    def _forward_mae_only(self, pts, vis=False):
        neighborhood, center = self.group_divider(pts)
        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B, _, C = x_vis.shape
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        x_rec = self.MAE_decoder(x_full, pos_full, N)
        B, M, C = x_rec.shape
        rebuild_points = (
            self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        )
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        mae_loss = self.loss_func(rebuild_points, gt_points)
        if vis:
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            return ret1, ret2, full_center, {
                "mae_loss": mae_loss.item(),
                "resa_loss": 0.0,
                "total_loss": mae_loss.item(),
            }
        return {
            "total_loss": mae_loss,
            "mae_loss": mae_loss,
            "resa_loss": torch.tensor(0.0, device=pts.device),
            "resa_weight": torch.tensor(0.0, device=pts.device),
        }

    def _forward_with_resa(self, pts, view_transform, vis=False, epoch=0):
        if epoch < self.resa_start_epoch:
            pts1, _ = view_transform(pts)
            return self._forward_mae_only(pts1, vis)
        # Fix FPS/KNN indices and the shared mask before either augmentation.
        _, original_center, group_indices = self.group_divider(pts, return_indices=True)
        shared_mask = self._generate_consistent_mask(original_center)
        pts1, pts2 = view_transform(pts)
        neighborhood1, center1 = self.group_divider(pts1, indices=group_indices)
        neighborhood2, center2 = self.group_divider(pts2, indices=group_indices)
        x_vis1, mask1 = self.MAE_encoder(neighborhood1, center1, shared_mask=shared_mask)
        x_vis2, mask2 = self.MAE_encoder(neighborhood2, center2, shared_mask=shared_mask)
        B, _, C = x_vis1.shape
        pos_emd_vis1 = self.decoder_pos_embed(center1[~mask1]).reshape(B, -1, C)
        pos_emd_mask1 = self.decoder_pos_embed(center1[mask1]).reshape(B, -1, C)
        _, N, _ = pos_emd_mask1.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full1 = torch.cat([x_vis1, mask_token], dim=1)
        pos_full1 = torch.cat([pos_emd_vis1, pos_emd_mask1], dim=1)
        x_rec1 = self.MAE_decoder(x_full1, pos_full1, N)
        B, M, C = x_rec1.shape
        rebuild_points1 = (
            self.increase_dim(x_rec1.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        )
        gt_points1 = neighborhood1[mask1].reshape(B * M, -1, 3)
        mae_loss = self.loss_func(rebuild_points1, gt_points1)
        resa_loss = torch.tensor(0.0, device=pts1.device)
        resa_weight = torch.tensor(0.0, device=pts1.device)
        if x_vis1.size(1) > 0 and x_vis2.size(1) > 0:
            if self.feature_detach:
                z1 = self.projection_head(x_vis1.detach())
                z2 = self.projection_head(x_vis2.detach())
            else:
                z1 = self.projection_head(x_vis1)
                z2 = self.projection_head(x_vis2)
            assert z1.size(1) == z2.size(1), f"Feature size mismatch: {z1.size(1)} vs {z2.size(1)}"
            resa_loss = self.resa_loss_fn(x_vis1, x_vis2, z1, z2)
            resa_weight = torch.tensor(self._get_adaptive_resa_weight(epoch), device=pts1.device)
        total_loss = mae_loss + resa_weight * resa_loss
        if vis:
            vis_points = neighborhood1[~mask1].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center1[~mask1].unsqueeze(1)
            full_rebuild = rebuild_points1 + center1[mask1].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            full_center = torch.cat([center1[mask1], center1[~mask1]], dim=0)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            return ret1, ret2, full_center, {
                "mae_loss": mae_loss.item(),
                "resa_loss": resa_loss.item(),
                "total_loss": total_loss.item(),
            }
        return {
            "total_loss": total_loss,
            "mae_loss": mae_loss,
            "resa_loss": resa_loss,
            "resa_weight": resa_weight,
        }

    def _generate_consistent_mask(self, center1):
        if self.MAE_encoder.mask_type == "rand":
            bool_masked_pos = self.MAE_encoder._mask_center_rand(center1, noaug=False)
        else:
            bool_masked_pos = self.MAE_encoder._mask_center_block(center1, noaug=False)
        return bool_masked_pos

    def _get_adaptive_resa_weight(self, epoch):
        if not hasattr(self.config, "resa_config"):
            return self.resa_lambda
        resa_config = self.config.resa_config
        adaptive_weight = resa_config.get("adaptive_weight", True)
        if not adaptive_weight:
            return self.resa_lambda
        start_epoch = self.resa_start_epoch
        warmup_epochs = resa_config.get("warmup_epochs", 120)
        if epoch < start_epoch:
            return 0.0
        # Linear warm-up to the nominal weight; keep that weight afterwards.
        if warmup_epochs > 0 and epoch < start_epoch + warmup_epochs:
            return self.resa_lambda * (epoch - start_epoch) / warmup_epochs
        return self.resa_lambda

    def get_encoder_features(self, pts, no_grad=False):
        if no_grad:
            with torch.no_grad():
                neighborhood, center = self.group_divider(pts)
                x_vis, mask = self.MAE_encoder(neighborhood, center, noaug=True)
        else:
            neighborhood, center = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center, noaug=True)
        return x_vis, center[~mask]
