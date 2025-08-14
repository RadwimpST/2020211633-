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

    device = torch.device("cuda" if args.cuda else "cpu")

    example_data, _ = next(iter(train_loader))  # example_data: [B, V, T]
    V = example_data.shape[1]
    T = example_data.shape[2]
    # 3 初始化自定义模型model_hierar
    # model = model_hierar(args).to(device)
    print("节点"+str(V)+"时间数"+str(T))
    model = model_hierar(args, seq_len=T, num_nodes=V).to(device)
    print(str(model))

    # 4 优化器选择
    # 当args==0时，选择SGD优化器（随机梯度下降）
    if args.use_sgd == 0:
        print("Use SGD")
        opt = optim.SGD(
            model.parameters(),  # 模型的所有参数
            lr=args.lr,  # 学习率
            momentum=0.9,  # 动量系数，用于加速SGD收敛
            nesterov=True,  # 启动 Nesterov 加速梯度
            weight_decay=0.0001)  # L2 正则化，用于防止过拟合
    else:
        print("Use Adam")
        opt = optim.Adam(
            model.parameters(),
            lr=args.lr,  # 建议 Adam 用 1e-3 左右
            betas=(0.9, 0.999),
            weight_decay=0.0001
            # amsgrad=True  # 如需可打开
        )
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

