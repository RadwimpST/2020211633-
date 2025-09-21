import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder
from omegaconf import DictConfig
from ..base import BaseModel

import numpy as np  # NEW
from .components.transformer_encoder import ModularTransformerEncoder  # NEW（你刚新增的编码器）


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(
        self,
        input_feature_size,
        input_node_num,
        hidden_size,
        output_node_num,
        pooling=True,
        orthogonal=True,
        freeze_center=False,
        project_assignment=True,
        # ===== NEW: modular 相关可选参数 =====
        use_modular: bool = True,
        modular_kwargs: dict = None,
        attn_mask_intra: torch.Tensor = None,
        attn_mask_inter: torch.Tensor = None,
        mods_index: torch.Tensor = None,
    ):
        super().__init__()
        self.use_modular = use_modular

        modular_kwargs = modular_kwargs or {}

        if use_modular:
            # 你在 components/transformer_encoder.py 里实现的两路编码器
            self.transformer = ModularTransformerEncoder(
                d_model=input_feature_size,
                nhead=modular_kwargs.get("nhead", 4),
                num_layers=1,                              # 与原先单层 InterpretableTransformerEncoder 对齐
                entmax_alpha=modular_kwargs.get("entmax_alpha", 1.3),
                film_dim=modular_kwargs.get("film_dim", 64),
                film_place=modular_kwargs.get("film_place", "pre_qkv"),
                fuse=modular_kwargs.get("fuse", "gate"),
                gate_granularity=modular_kwargs.get("gate_granularity", "global"),
                dropout=modular_kwargs.get("dropout", 0.1),
            )
            assert mods_index is not None and attn_mask_intra is not None and attn_mask_inter is not None
            self.register_buffer("mods_index", mods_index.to(torch.long))  # long 索引
            self.register_buffer("attn_mask_intra", attn_mask_intra.to(torch.float))  # additive mask: 允许=0, 禁止=-1e9
            self.register_buffer("attn_mask_inter", attn_mask_inter.to(torch.float))
        else:
            # 原版编码器（保持不变）
            self.transformer = InterpretableTransformerEncoder(
                d_model=input_feature_size, nhead=4,
                dim_feedforward=hidden_size,
                batch_first=True
            )

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size * input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        if self.use_modular:
            x = self.transformer(
                x,
                attn_mask_intra=self.attn_mask_intra,
                attn_mask_inter=self.attn_mask_inter,
                mods=self.mods_index
            )
        else:
            x = self.transformer(x)

        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        # 兼容：如果 modular 编码器没实现可视化接口，就返回 None
        return getattr(self.transformer, "get_attention_weights", lambda: None)()

    def loss(self, assignment):
        return self.dec.loss(assignment)



class BrainNetworkTransformer(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        # ===== NEW: modular 编码器所需的模块映射与掩码 =====
        self.use_modular = getattr(config.model, "encoder", "vanilla") == "modular"
        if self.use_modular:
            mods = np.load(config.model.module_map_path).astype(np.int64)  # [N], 1..K
            assert mods.shape[0] == config.dataset.node_sz, \
                f"module map length={mods.shape[0]} != node_sz={config.dataset.node_sz}"
            mods_t = torch.from_numpy(mods)

            # intra：同模块允许（含自环）；inter：跨模块允许（禁自环）
            M_intra = (mods_t[:, None] == mods_t[None, :]).float()
            M_inter = 1.0 - M_intra
            M_intra.fill_diagonal_(1.0)
            M_inter.fill_diagonal_(0.0)

            # additive mask（允许=0，禁止=-1e9）
            attn_mask_intra = (1.0 - M_intra) * (-1e9)
            attn_mask_inter = (1.0 - M_inter) * (-1e9)

            # # 缓存为 buffer（随 .to(device) 一起迁移）
            # self.register_buffer("attn_mask_intra", attn_mask_intra)
            # self.register_buffer("attn_mask_inter", attn_mask_inter)
            # self.register_buffer



        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            use_mod = bool(self.use_modular and index == 0)  # 只在第一层使用两路编码器
            self.attention_list.append(
                TransPoolingEncoder(
                    input_feature_size=forward_dim,
                    input_node_num=in_sizes[index],
                    hidden_size=1024,
                    output_node_num=size,
                    pooling=do_pooling[index],
                    orthogonal=config.model.orthogonal,
                    freeze_center=config.model.freeze_center,
                    project_assignment=config.model.project_assignment,
                    # ===== NEW: 仅第一层传 modular 相关参数 =====
                    use_modular=use_mod,
                    modular_kwargs=dict(
                        nhead=4,
                        entmax_alpha=getattr(config.model, "entmax", {}).get("alpha", 1.3),
                        film_dim=getattr(config.model, "film", {}).get("dim", 64),
                        film_place=getattr(config.model, "film", {}).get("place", "pre_qkv"),
                        fuse=getattr(config.model, "fuse", "gate"),
                        gate_granularity=getattr(getattr(config.model, "gate", {}), "granularity", "global"),
                        dropout=0.1,
                    ) if use_mod else None,
                    # attn_mask_intra=self.attn_mask_intra if use_mod else None,
                    # attn_mask_inter=self.attn_mask_inter if use_mod else None,
                    # mods_index=self.mods_index if use_mod else None,
                    attn_mask_intra=attn_mask_intra,
                    attn_mask_inter=attn_mask_inter,
                    mods_index=mods_t,
                )
            )


        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):

        bz, _, _, = node_feature.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
