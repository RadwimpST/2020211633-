
import os
import math
import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import DataLoadAdni
from model_modular import ModelModular
from loss_utils import (
    ce_loss, balance_loss, sharp_loss,
    intra_compact_loss, inter_separate_loss, empty_module_penalty,
    schedule_linear, schedule_cosine
)


# --------------------------- utils -------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_classes(dataset) -> int:
    """Infer number of classes from dataset labels."""
    # sample all labels (labels are small compared to data)
    labels = []
    for i in range(len(dataset)):
        labels.append(int(dataset[i][1]))
    classes = int(max(labels)) + 1 if labels else 2
    return classes


def build_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    train_set = DataLoadAdni(args.choose_data, args.partroi, args.train_partition, args.fold)
    eval_set = DataLoadAdni(args.choose_data, args.partroi, args.eval_partition, args.fold)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return train_loader, eval_loader, train_set, eval_set


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / max(total, 1)


# --------------------------- training ----------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch, args):
    model.train()
    total = 0
    sum_loss = 0.0
    sum_acc = 0.0
    # NEW: 分量损失的累计（加权后的值）
    comp_sums = dict(ce=0.0, bal=0.0, sharp=0.0, intra=0.0, inter=0.0, empty=0.0)

    for x, y in loader:
        # x: [B, V, T], y: [B]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        # schedule tau by epoch (simple; can be by step if desired)
        if args.tau_schedule == 'linear':
            tau = schedule_linear(epoch, start=args.tau_start, end=args.tau_end, total_steps=max(args.epochs-1, 1))
        else:
            tau = schedule_cosine(epoch, start=args.tau_start, end=args.tau_end, total_steps=max(args.epochs-1, 1))
        model.set_tau(tau)

        assign_source = 'soft' if epoch < args.assign_warmup_epochs else 'hard'
        detach_mask = bool(args.detach_mask)

        out = model(x, assign_source=assign_source, detach_mask=detach_mask, return_tokens=False)

        logits = out['logits']
        H = out['H']
        S = out['assign_soft']
        P = out['alloc_extras']['prototypes']
        usage = out['alloc_extras']['usage_per_k']

        # --- losses ---
        L_cls = ce_loss(logits, y, label_smoothing=0.0, class_weights=None, reduction='mean')

        L_balance = balance_loss(usage) * args.lambda_balance
        L_sharp = sharp_loss(S) * args.lambda_sharp
        L_intra = intra_compact_loss(H, S, P, metric='l2', normalize=False, detach_P=False) * args.lambda_intra
        L_inter = inter_separate_loss(P, mode='orth', normalize=True) * args.lambda_inter

        L_empty = torch.tensor(0.0, device=device)

        # NEW: 按 batch 累计（注意乘以 bsz，方便最后求平均）
        bsz = y.size(0)
        comp_sums['ce'] += L_cls.item() * bsz
        comp_sums['bal'] += L_balance.item() * bsz
        comp_sums['sharp'] += L_sharp.item() * bsz
        comp_sums['intra'] += L_intra.item() * bsz
        comp_sums['inter'] += L_inter.item() * bsz
        comp_sums['empty'] += L_empty.item() * bsz

        if args.lambda_empty > 0.0:
            # use soft counts to allow gradients to flow to the allocator
            soft_count = S.sum(dim=1)  # [B,K]
            L_empty = empty_module_penalty(soft_count, min_count=args.min_count_per_module, mode='hinge') * args.lambda_empty

        loss = L_cls + L_balance + L_sharp + L_intra + L_inter + L_empty

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)

        optimizer.step()

        # metrics
        bsz = y.size(0)
        total += bsz
        sum_loss += loss.item() * bsz
        sum_acc += accuracy_from_logits(logits, y) * bsz

    #return sum_loss / max(total, 1), sum_acc / max(total, 1)
    # NEW: 计算分量的 epoch 平均并一起返回
    for k in comp_sums:
        comp_sums[k] = comp_sums[k] / max(total, 1)
    return sum_loss / max(total, 1), sum_acc / max(total, 1), comp_sums


@torch.no_grad()
def evaluate(model, loader, device, epoch, args):
    model.eval()
    total = 0
    sum_loss = 0.0
    sum_acc = 0.0

    comp_sums = dict(ce=0.0, bal=0.0, sharp=0.0, intra=0.0, inter=0.0, empty=0.0)

    # fix tau & use hard assign at eval
    model.set_tau(args.tau_eval)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        out = model(x, assign_source='hard', detach_mask=True, return_tokens=False)

        logits = out['logits']
        H = out['H']
        S = out['assign_soft']
        P = out['alloc_extras']['prototypes']
        usage = out['alloc_extras']['usage_per_k']

        L_cls = ce_loss(logits, y, label_smoothing=0.0, class_weights=None, reduction='mean')
        L_balance = balance_loss(usage) * args.lambda_balance
        L_sharp = sharp_loss(S) * args.lambda_sharp
        L_intra = intra_compact_loss(H, S, P, metric='l2', normalize=False, detach_P=False) * args.lambda_intra
        L_inter = inter_separate_loss(P, mode='orth', normalize=True) * args.lambda_inter
        bsz = y.size(0)
        comp_sums['ce'] += L_cls.item() * bsz
        comp_sums['bal'] += L_balance.item() * bsz
        comp_sums['sharp'] += L_sharp.item() * bsz
        comp_sums['intra'] += L_intra.item() * bsz
        comp_sums['inter'] += L_inter.item() * bsz
        # comp_sums['empty'] 保持 0.0

        loss = L_cls + L_balance + L_sharp + L_intra + L_inter

        bsz = y.size(0)
        total += bsz
        sum_loss += loss.item() * bsz
        sum_acc += accuracy_from_logits(logits, y) * bsz

    #return sum_loss / max(total, 1), sum_acc / max(total, 1)
    # NEW
    for k in comp_sums:
        comp_sums[k] = comp_sums[k] / max(total, 1)
    return sum_loss / max(total, 1), sum_acc / max(total, 1), comp_sums

# NEW: 50个epoch保存“每个受试者的节点分配情况”
@torch.no_grad()
def save_allocations(model, loader, device, k_modules: int, out_path: str, epoch_idx: int):
    """
    迭代 loader（不打乱），对每个受试者输出：
    受试者N-<标签>：1：[节点列表]   2：[节点列表] ...
    模块与节点索引均为 1-based；每 50 个 epoch 追加一段。
    """
    model.eval()
    subj_counter = 0
    label_names = {0: "CN", 1: "Patient"}  # 你可以按需要扩展/修改

    lines = [f"=== Epoch {epoch_idx:03d} ==="]
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        out = model(x, assign_source='hard', detach_mask=True, return_tokens=False)
        Z = out['assign_hard']               # [B,V,K] one-hot（ST 前向）
        assign_ids = Z.argmax(dim=-1)        # [B,V]  每个节点所属模块（0~K-1）

        B, V = assign_ids.shape
        for b in range(B):
            subj_counter += 1
            label_id = int(y[b].item())
            label_txt = label_names.get(label_id, str(label_id))

            # 分模块收集该受试者的节点索引（1-based）
            parts = []
            for k in range(k_modules):
                nodes_0 = (assign_ids[b] == k).nonzero(as_tuple=False).view(-1).tolist()
                nodes_1 = [idx + 1 for idx in nodes_0]  # 转为 1-based
                parts.append(f"{k+1}：{nodes_1}")
            line = f"受试者{subj_counter}-{label_txt}：" + "   ".join(parts)
            lines.append(line)

    # 追加写入
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train modular fMRI model (TimeEncoder + Allocator + MFeature).")

    # data & loader
    parser.add_argument('--choose_data', type=str, default='ADNI2')
    parser.add_argument('--partroi', type=int, default=116)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--train_partition', type=str, default='train')
    parser.add_argument('--eval_partition', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)

    # model
    parser.add_argument('--k_modules', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--d_embed', type=int, default=16)
    parser.add_argument('--use_module_embed', type=int, default=1)

    # training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--eval_interval', type=int, default=1)

    # allocator schedule / assignment strategy
    parser.add_argument('--tau_start', type=float, default=1.8)
    parser.add_argument('--tau_end', type=float, default=0.5)
    parser.add_argument('--tau_schedule', type=str, choices=['linear','cosine'], default='cosine')
    parser.add_argument('--tau_eval', type=float, default=0.5)
    parser.add_argument('--assign_warmup_epochs', type=int, default=3)
    parser.add_argument('--detach_mask', type=int, default=1)  # 1=True, 0=False

    # losses (lambdas)
    parser.add_argument('--lambda_balance', type=float, default=0.2)
    parser.add_argument('--lambda_sharp', type=float, default=0.1)
    parser.add_argument('--lambda_intra', type=float, default=0.5)
    parser.add_argument('--lambda_inter', type=float, default=0.05)
    parser.add_argument('--lambda_empty', type=float, default=0.0)
    parser.add_argument('--min_count_per_module', type=float, default=1.0)

    # gradient clipping (switchable): 0 -> disabled
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print('using device:', device, 'cuda_available:', torch.cuda.is_available())
    if device.type == 'cuda':
        print('gpu name:', torch.cuda.get_device_name(torch.cuda.current_device()))
    # data
    train_loader, eval_loader, train_set, eval_set = build_dataloaders(args)
    # infer num_classes from train_set
    num_classes = count_classes(train_set)

    # model
    # Need seq_len and num_nodes from a sample
    sample_x, sample_y = train_set[0]
    V, T = sample_x.shape
    model = ModelModular(
        seq_len=T,
        num_nodes=V,
        num_classes=num_classes,
        k_modules=args.k_modules,
        d_model=args.d_model,
        d_embed=args.d_embed,
        use_module_embed=bool(args.use_module_embed),
        encoder_kwargs=dict(),      # keep defaults from TimeEncoder
        allocator_kwargs=dict(),    # keep defaults from Allocator
        mfeature_kwargs=dict(),     # defaults
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    best_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        #tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, args)
        tr_loss, tr_acc, tr_comp = train_one_epoch(model, train_loader, optimizer, device, epoch, args)

        if (epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs:
            # val_loss, val_acc = evaluate(model, eval_loader, device, epoch, args)
            val_loss, val_acc, val_comp = evaluate(model, eval_loader, device, epoch, args)
            #print(f"[Epoch {epoch+1:03d}] "
            #      f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            #      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            print(
                f"[Epoch {epoch + 1:03d}] "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                f"(CE={tr_comp['ce']:.4f}, bal={tr_comp['bal']:.4f}, sharp={tr_comp['sharp']:.4f}, "
                f"intra={tr_comp['intra']:.4f}, inter={tr_comp['inter']:.4f}, empty={tr_comp['empty']:.4f}) | "
                f"val_loss={val_loss:.4f} "
                f"(CE={val_comp['ce']:.4f}, bal={val_comp['bal']:.4f}, sharp={val_comp['sharp']:.4f}, "
                f"intra={val_comp['intra']:.4f}, inter={val_comp['inter']:.4f}) "
                f"train_acc={tr_acc:.4f} val_acc={val_acc:.4f}"
            )
            # NEW: 每 50 个 epoch 保存一次分配情况
            if (epoch + 1) % 50 == 0:
                alloc_path = str(save_dir / "allocate.txt")
                save_allocations(model, eval_loader, device, args.k_modules, alloc_path, epoch + 1)

            if val_acc > best_acc:
                best_acc = val_acc
                ckpt = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'args': vars(args),
                }
                torch.save(ckpt, save_dir / 'model_best.pt')

    # save last
    ckpt = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': args.epochs - 1,
        'best_acc': best_acc,
        'args': vars(args),
    }
    torch.save(ckpt, save_dir / 'last.pt')
    print(f"Training finished. Best val acc: {best_acc:.4f}. Checkpoints at: {save_dir}")


if __name__ == "__main__":
    main()
