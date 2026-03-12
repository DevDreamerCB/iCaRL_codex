import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import math


class TaskEmbeddingAdapter(nn.Module):
    def __init__(self, emb_size, adapter_dim=32, num_tasks=3, dropout=0.1, start_task=0):
        super().__init__()
        self.num_tasks = num_tasks
        self.start_task = start_task
        self.current_task = 0
        self.norm = nn.LayerNorm(emb_size)
        self.down = nn.ModuleList([nn.Linear(emb_size, adapter_dim) for _ in range(num_tasks)])
        self.up = nn.ModuleList([nn.Linear(adapter_dim, emb_size) for _ in range(num_tasks)])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        for up in self.up:
            nn.init.zeros_(up.weight)
            nn.init.zeros_(up.bias)

    def set_current_task(self, task_id):
        self.current_task = max(0, min(int(task_id), self.num_tasks - 1))

    def forward(self, x):
        if self.current_task < self.start_task:
            return x
        x_norm = self.norm(x)
        hidden = self.down[self.current_task](x_norm)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        return x + self.up[self.current_task](hidden)

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim=128, num_channels=45):
        super().__init__()

        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 25), stride=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(self.num_channels, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(128)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(0.5)

        self.projection = nn.Sequential(
            nn.Conv2d(128, embed_dim, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.chan_embed = nn.Embedding(45, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.unsqueeze(1)  # (B, 1, C, T)
        B, _ ,C, T = x.size()
        x = self.conv1(x)
        
        x = self.conv2(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=8, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size,dropout=0.5):
        super().__init__(*[TransformerEncoderBlock(emb_size,drop_p=dropout) for _ in range(depth)])

class decoder(nn.Module):  
    def __init__(self, emb_size=64, depth=2, pretrain=None,**kwargs):
        super().__init__()
        self.transformer = TransformerEncoder(depth, emb_size)
    def forward(self, x):
        x = self.transformer(x)      # [batch_size, seq_length, emb_size]
        return x

class decoder_fft(nn.Module): 
    def __init__(self, emb_size=64, depth=2, pretrain=None,**kwargs):
        super().__init__()
        self.transformer = TransformerEncoder(depth, emb_size)
        self.pro= nn.Linear(emb_size, 3*2)  
    def forward(self, x):
        out = self.transformer(x)      # [batch_size, seq_length, emb_size]
        out = self.pro(torch.mean(out, dim=1))  # [batch_size, seq_length, 3*2]
        return out

class mlm_mask(nn.Module):  
    def __init__(self, emb_size=128, depth=6, n_classes=2,mask_ratio=0.5, pretrain=None,pretrainmode=False,
                 use_task_adapter=False, adapter_dim=32, num_tasks=3, adapter_start_task=0):
        super().__init__()
        self.pretrainmode = pretrainmode
        self.embedding = PatchEmbedding(embed_dim=emb_size)
        self.transformer = TransformerEncoder(depth, emb_size,dropout=0.5)
        self.clshead = nn.Linear(emb_size,n_classes)
        self.mask_ratio = mask_ratio
        self.feature_dim = emb_size
        self.use_task_adapter = use_task_adapter
        self.current_task = 0
        self.task_adapter = TaskEmbeddingAdapter(
            emb_size,
            adapter_dim=adapter_dim,
            num_tasks=num_tasks,
            start_task=adapter_start_task,
        ) if use_task_adapter else None
        if pretrain is not None:
            self.init_from_pretrained(pretrain)
        
        if self.pretrainmode:
            self.mask_token = nn.Parameter(torch.randn(1, 1, emb_size))
            self.decoder = decoder(emb_size=emb_size, depth=2)

    def random_masking(self, x, mask_ratio=0.5):
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
      
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask_tokens = self.mask_token.repeat(B, L - len_keep, 1)
        x_masked = torch.cat([x_masked, mask_tokens], dim=1)

        x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.zeros([B, L], device=x.device)
        mask[:, :len_keep] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward(self, x):
        original_x = self.embedding(x) 
        if self.task_adapter is not None:
            original_x = self.task_adapter(original_x)

        if self.pretrainmode:

            x_masked, mask, ids_restore = self.random_masking(original_x,mask_ratio=self.mask_ratio)

            encoded = self.transformer(x_masked)
 
            reconstructed = self.decoder(encoded)

            cls_output = self.clshead(torch.mean(encoded, dim=1))
            
            return cls_output, original_x, reconstructed, None
        else:
            transformed = self.transformer(original_x)
            pooled = torch.mean(transformed, dim=1)
            # cls_output = self.clshead(pooled)
            # return pooled, cls_output
            return pooled

    def set_current_task(self, task_id):
        self.current_task = int(task_id)
        if self.task_adapter is not None:
            self.task_adapter.set_current_task(task_id)

    def init_from_pretrained(self, pretrained_path, freeze_encoder=False, strict=True):
        pretrained_dict = torch.load(pretrained_path)
        
        model_dict = self.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        
        self.load_state_dict(model_dict, strict=strict)
        
        if freeze_encoder:
            for name, param in self.named_parameters():
                if 'embedding' in name or 'transformer' in name:
                    param.requires_grad = False
                # if 'transformer' in name:
                #     param.requires_grad = False
        
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from pretrained model")

if __name__ == "__main__":
    # 1. 加载预训练 dict（允许 map_location）
    pretrained_path = '/data1/bochen/MIRepNet/weight/MIRepNet.pth'
    model = mlm_mask(emb_size=256, depth=6, n_classes=3, mask_ratio=0.5, pretrain=pretrained_path, pretrainmode=False)
    
    freeze_encoder=False
    strict=True
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')

    model_dict = model.state_dict()

    x = torch.randn((32, 1, 45, 1001))
    print(model(x).shape)
    input(' ')
    # 2. 筛选能匹配的键（键名相同且形状相同）
    matched = {}
    shape_mismatch = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                matched[k] = v
            else:
                shape_mismatch[k] = (v.shape, model_dict[k].shape)
        else:
            # 未在 model 中找到该键
            pass

    # 3. 未被预训练覆盖的模型键
    unmatched_model_keys = [k for k in model_dict.keys() if k not in matched]

    print(f"Pretrained file keys: {len(pretrained_dict)}")
    print(f"Model keys total: {len(model_dict)}")
    print(f"Matched keys to be loaded: {len(matched)}")
    print(f"Shape-mismatched keys in pretrained: {len(shape_mismatch)}")
    print(f"Model keys that remain from random init: {len(unmatched_model_keys)}")
    print()

    if len(matched) > 0:
        print("=== 示例：被匹配并将被加载的前 50 个键 ===")
        for i, k in enumerate(list(matched.keys())[:50]):
            print(f"{i+1:03d}: {k} -> shape {matched[k].shape}")
        print()

    if len(shape_mismatch) > 0:
        print("=== 形状不匹配的键（预训练形状 -> 模型形状） ===")
        for k, (ps, ms) in list(shape_mismatch.items())[:20]:
            print(f"{k}: {ps} -> {ms}")
        print()

    if len(unmatched_model_keys) > 0:
        print("=== 模型中但未被预训练覆盖的键（示例） ===")
        for k in unmatched_model_keys[:50]:
            print(k)
        print()

    # 4. 现在把 matched 更新到 model_dict 并 load（和你的实现一致）
    model_dict.update(matched)
    model.load_state_dict(model_dict, strict=strict)

    # 5. 如果需要冻结 encoder
    if freeze_encoder:
        for name, param in model.named_parameters():
            if 'embedding' in name or 'transformer' in name:
                param.requires_grad = False

    # 6. 输出参数可学习情况
    print("=== 参数 requires_grad 列表 ===")
    for name, param in model.named_parameters():
        print(f"{name}  shape={tuple(param.shape)}  requires_grad={param.requires_grad}")
