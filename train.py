from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data import DataLoadAdni
from model_hierar import model_hierar
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import time

def dump_assignments(model, dataset, indices, epoch, device, out_dir="./assign"):
    """
    对给定 indices 的样本导出两类文件到 ./assign/epoch_XXXX/：
      1) sub-<idx>_S.csv     : (V x K) 软分配矩阵（每行和≈1）
      2) sub-<idx>_top5.csv  : (K x 5) 各模块 Top5 节点的索引（0-based）
    """
    import numpy as np
    import os
    os.makedirs(out_dir, exist_ok=True)
    ep_dir = os.path.join(out_dir, f"epoch_{epoch:04d}")
    os.makedirs(ep_dir, exist_ok=True)

    was_training = model.training
    model.eval()

    for idx in indices:
        data, _ = dataset[idx]                 # data: (V,T) per your DataLoadAdni
        if isinstance(data, np.ndarray):
            data_t = torch.from_numpy(data).float()
        else:
            data_t = data.float()
        data_t = data_t.unsqueeze(0).to(device)  # (1,V,T)

        S = model.infer_assign(data_t).cpu().numpy()[0]  # (V,K)

        # 1) 保存 S 矩阵
        np.savetxt(os.path.join(ep_dir, f"sub-{idx:05d}_S.csv"),
                   S, delimiter=",", fmt="%.6f")

        # 2) 保存 Top5 节点（按列取 top5 行索引）
        V, K = S.shape
        top5 = np.zeros((K, 5), dtype=int)
        k5 = min(5, V)
        for k in range(K):
            top_idx = np.argsort(-S[:, k])[:k5]
            # 若 V<5，用 -1 填充剩余列（也可以保持更短列；按你需求）
            if k5 < 5:
                pad = np.full((5 - k5,), -1, dtype=int)
                top5[k] = np.concatenate([top_idx, pad], axis=0)
            else:
                top5[k] = top_idx
        np.savetxt(os.path.join(ep_dir, f"sub-{idx:05d}_top5.csv"),
                   top5, delimiter=",", fmt="%d")

    # 还原训练状态
    if was_training:
        model.train()


import math

# 退火函数
def tau_schedule(epoch_idx, args):
    """epoch_idx 从 1 开始；达到下限后保持 tau_min。"""
    e = max(0, epoch_idx - args.tau_warmup_epochs)
    if args.tau_schedule == 'linear':
        # 线性: 1.8 → 0.5
        p = min(1.0, e / max(1, args.tau_anneal_epochs))
        return args.tau_init + (args.tau_min - args.tau_init) * p
    elif args.tau_schedule == 'cosine':
        # 余弦: 前期更平滑
        p = min(1.0, e / max(1, args.tau_anneal_epochs))
        return float(args.tau_min + 0.5 * (args.tau_init - args.tau_min) * (1 + math.cos(math.pi * p)))
    elif args.tau_schedule == 'exp':
        # 指数衰减（gamma≈0.9）
        gamma = (args.tau_min / args.tau_init) ** (1.0 / max(1, args.tau_anneal_epochs))
        return max(args.tau_min, args.tau_init * (gamma ** e))
    elif args.tau_schedule == 'step':
        # 每 N=5 个 epoch 乘个系数 0.85
        N, factor = 5, 0.85
        steps = max(0, (e // N))
        return max(args.tau_min, args.tau_init * (factor ** steps))


# 1 init这个函数检查是否存在用于保存日志和模型的目录。如果没有，就创建这些目录。
def _init_():
    if not os.path.exists('log'):
        os.makedirs('log')
    if not os.path.exists('log/' + args.exp_name):
        os.makedirs('log/' + args.exp_name)
    if not os.path.exists('log/' + args.exp_name + '/' + 'models'):
        os.makedirs('log/' + args.exp_name + '/' + 'models')


def train(args, fold):
    # 1参数：args：超参数 fold：交叉验证折数

    # python  train.py --partroi 270 --num_pooling 1 --assign_ratio 0.35 --assign_ratio_1 0.35 --mult_num 8

    # 2数据加载：
    # fold表示当前的折数，shuffle=True表示打乱顺序，drop_last=True表示如果最后一个批次样本不足一个批次，则丢弃
    train_loader = DataLoader(
        DataLoadAdni(partition='train', partroi=args.partroi, fold=fold + 1, choose_data=args.data_choose),
        num_workers=0,
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        DataLoadAdni(partition='test', partroi=args.partroi, fold=fold + 1, choose_data=args.data_choose),
        num_workers=0,
        batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    # 固定 10 个锚定样本（从训练集里取；如需验证集就从 val_loader.dataset 取）
    monitor_dataset = train_loader.dataset
    num_pick = min(10, len(monitor_dataset))
    monitor_indices = list(range(num_pick))  # 或者你自定义的 ID 列表

    device = torch.device("cuda" if args.cuda else "cpu")

    example_data, _ = next(iter(train_loader))  # example_data: [B, V, T]
    V = example_data.shape[1]
    T = example_data.shape[2]
    # 3 初始化自定义模型model_hierar
    model = model_hierar(args).to(device)
    print("节点"+str(V)+"时间数"+str(T))
    #model = model_hierar(args, seq_len=T, num_nodes=V).to(device)
    print(str(model))

    # 4 优化器选择
    # 4 优化器选择
    if args.use_sgd == 0:
        print("Use SGD")
        opt = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.0001)
    elif args.use_sgd == 1:
        print("Use Adam")
        opt = optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0001)
    elif args.use_sgd == 2:
        print("Use AdamW")
        # 注意：AdamW 在 PyTorch 中是单独的类
        opt = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0001)
    else:
        # 增加一个默认选项，处理未知的 args.use_sgd 值
        raise ValueError("Invalid optimizer choice. Please use 0 for SGD, 1 for Adam, or 2 for AdamW.")
    loss_entro = nn.CrossEntropyLoss().cuda()
    best_test_acc = 0  # 用于记录训练过程中的最佳模型

    for epoch in range(args.epochs):
        # 当前epoch的损失：train_loss 样本数量计数器：count
        train_loss = 0.0
        count = 0.0
        # 训练模式，启用dropout等训练时才有的操作
        model.train()
        # train_pred和_true存储预测值与真实值
        train_pred = []
        train_true = []
        idx = 0
        total_time = 0.0

        tau = tau_schedule(epoch, args)
        model.allocator.set_tau(float(tau))

        # 如果想记录：
        print(f"[epoch {epoch}] tau={tau:.3f}")

        for data, label in train_loader:
            # 从loader得到数据
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[0]
            start_time = time.time()
            # 前向传播与损失
            logits = model(data)  # data输入模型，得到 预测值logits
            logits = logits.squeeze(1)  # 去除预测值维度为1的部分
            label = label.to(torch.float32)  # 兼容损失函数
            loss = loss_entro(logits, label.long()).cuda()  # 计算交叉熵

            # 反向传播
            opt.zero_grad()
            loss.backward()
            opt.step()

            end_time = time.time()
            total_time += (end_time - start_time)

            value11, preds = torch.max(logits.data, 1)  # 预测类别：选择维度1上最大的值作为预测值

            count += batch_size
            train_loss += loss.item() * batch_size  # 累计训练损失
            # 将标签与预测值从GPU转移到CPU并转换为numpy数组，方便后续计算准确率
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1
        print('train total time is', total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f' % (
        epoch, train_loss * 1.0 / count, metrics.accuracy_score(train_true, train_pred))
        print(outstr)
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[0]
            start_time = time.time()
            logits = model(data)
            logits = logits.squeeze(1)
            label = label.to(torch.float32)
            loss = loss_entro(logits, label.long()).cuda()
            value22, preds = torch.max(logits.data, 1)
            end_time = time.time()
            total_time += (end_time - start_time)
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        print('test total time is', total_time)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        test_acc = metrics.accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f,  test acc: %.6f' % (epoch, test_loss * 1.0 / count, test_acc)
        print(outstr)

        # 每 50 个 epoch 导出一次(50, 100, 150, ...)
        if (epoch + 1) % 50 == 0:
            dump_assignments(model,
                             monitor_dataset,
                             monitor_indices,
                             epoch + 1,
                             device,
                             out_dir="./assign")

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            state = {
                'epoch': epoch,
                'acc': best_test_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }
            torch.save(state, 'log/%s/models/model.t7' % args.exp_name)
        outstr = 'best_epoch: %d,best_acc: %.6f' % (best_epoch, best_test_acc)
        print(outstr)
    return best_test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HFBN')
    parser.add_argument('--exp_name', type=str, default='train', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=30, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=299, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=int, default=0,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N')
    parser.add_argument('--Num', type=int, default=0, help=' ')
    parser.add_argument('--adni', type=int, default=2, choices=[2, 3])
    parser.add_argument('--kernel', type=int, default=9)
    parser.add_argument('--partroi', type=int, default=270)
    parser.add_argument('--log_dir', type=str, default='output', help='experiment root')
    parser.add_argument('--Gbias', type=bool, default=False, help='if bias ')
    parser.add_argument('--num_pooling', type=int, default=1, help=' ')
    parser.add_argument('--embedding_dim', type=int, default=90, help=' ')
    parser.add_argument('--assign_ratio', type=float, default=0.35, help=' ')
    parser.add_argument('--assign_ratio_1', type=float, default=0.35, help=' ')
    parser.add_argument('--mult_num', type=int, default=8, help=' ')
    parser.add_argument('--data_choose', type=str, default='adni2', help='choose model:adni2 or adni3')
    parser.add_argument('--fold_list', default=[3], help='fold = 0,1,2,3,4')
    # 新增：控制 num_pooling==0 的两个开关
    parser.add_argument('--if_norm', type=int, default=1, choices=[0, 1],
                        help='(only when num_pooling==0) 1: use conv1->BN->ReLU->Dropout after attention; 0: skip')
    parser.add_argument('--if_q', type=str, default='mean',
                        choices=['mean', 'q', 'cls'],
                        help="(only when num_pooling==0) readout: 'mean' for global mean, 'q' for global query attention, 'cls' for CLS token")
    parser.add_argument('--num_modules', type=int, default=4, help=' ')

    # 退火部分
    parser.add_argument('--tau_init', type=float, default=1.8)
    parser.add_argument('--tau_min', type=float, default=0.5)
    parser.add_argument('--tau_anneal_epochs', type=int, default=15)  # 退火总历元数
    parser.add_argument('--tau_warmup_epochs', type=int, default=0)  # 可设 0
    parser.add_argument('--tau_schedule', type=str, default='linear', choices=['linear', 'cosine', 'exp', 'step'])

    allaccu = []
    all_result = []
    args = parser.parse_args()
    # 3. 设置实验目录和模型路径
    fold_list = args.fold_list  # 获取交叉验证的折数列表
    for i in fold_list:  # 循环遍历每个折数（交叉验证的每个划分）
        i = int(i)
        fold = i
        args = parser.parse_args()

        args.exp_name = str(args.log_dir) + '/' + 'adni' + str(args.adni) + '_roi' + str(args.partroi) + '_pool' + str(
            args.num_pooling) + '_fold' + str(i + 1) + '/' + args.exp_name
        args.model_path = str(args.log_dir) + '/' + 'adni' + str(args.adni) + '_roi' + str(
            args.partroi) + '_pool' + str(args.num_pooling) + '_fold' + str(i + 1) + '/models/model.t7'
        _init_()
        args.cuda = not args.no_cuda and torch.cuda.is_available()

        if args.cuda:
            print(
                'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(
                    torch.cuda.device_count()) + ' devices')
        else:
            print('Using CPU')
        if not args.eval:
            accu = train(args, fold)
        allaccu.append(accu)

    array = np.array(allaccu)
    print('Accuracy summary:')
    print(np.mean(array, axis=0))

    with open('log/adni_{}_roi_{}_log.txt'.format(args.adni, args.partroi), 'a') as f:
        print('*************:', file=f)
        print('adni' + str(args.adni) + '_roi' + str(args.partroi) + '_lr' + str(args.lr) + '_BS' + str(
            args.batch_size) + '_pool' + str(args.num_pooling) + '_rate' + str(args.assign_ratio) + ':', file=f)
        print(array, file=f)
        print('Accuracy summary:', file=f)
        print(np.mean(array, axis=0), file=f)

