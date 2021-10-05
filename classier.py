# -*- coding: utf-8 -*-

"""
@date: 2020/3/1 上午9:54
@file: finetune.py
@author: zj
@description:
"""

import os
import copy
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from data.classifer_dataset import CustomClassifierDataset
from data.finetune_sample import CustomBatchSampler
from data.HardNegativeMining import CustomHardNegativeMiningDataset

def hinge_loss(outputs, labels):
    """
    折页损失计算
    :param outputs: 大小为(N, num_classes)
    :param labels: 大小为(N)
    :return: 损失值
    """
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T

    # 最大间隔
    margin = 1.0
    margins = outputs - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)

    # # 正则化强度
    # reg = 1e-3
    # loss += reg * torch.sum(weight ** 2)

    return loss

def load_data(data_root_dir):
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loaders = {}
    data_sizes = {}

    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name)
        data_set = CustomClassifierDataset(data_dir, transform=transform)
        if name == 'train':
            positive_list = data_set.get_positives()
            negative_list = data_set.get_negatives()
            #随机抽取和正样本的一样数量的副样本
            init_negative_idxs = random.sample(range(len(negative_list)), len(positive_list))
            #选出样本编号以及剩余样本
            init_negative_list = [negative_list[idx] for idx in range(len(negative_list)) if idx in init_negative_idxs]
            remain_negative_list = [negative_list[idx] for idx in range(len(negative_list))
                                    if idx not in init_negative_idxs]
            #设置负样本到dataset中,然后把剩余的保存起来
            data_set.set_negative_list(init_negative_list)
            data_loaders['remain'] = remain_negative_list
        data_sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(), 32, 96)
        data_loader = DataLoader(data_set, batch_size=128, sampler=data_sampler, num_workers=8, drop_last=True)

        data_loaders[name] = data_loader
        data_sizes[name] = data_sampler.__len__()

    return data_loaders, data_sizes
def add_hard_negatives(hard_negative_list, negative_list, add_negative_list):
    for item in hard_negative_list:
        if len(add_negative_list) == 0:
            # 第一次添加负样本
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))
        if list(item['rect']) not in add_negative_list:
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))
def get_hard_negatives(preds, cache_dicts):

    fp_mask = preds == 1

    fp_rects = cache_dicts['rect'][fp_mask].numpy()
    fp_image_ids = cache_dicts['image_id'][fp_mask].numpy()

    hard_negative_list = [{'rect': fp_rects[idx], 'image_id': fp_image_ids[idx]} for idx in range(len(fp_rects))]

    return hard_negative_list
def train_model(data_loaders, model, criterion,lr_scheduler, optimizer, num_epochs=30, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        ammu=4

        # 最小化数据
        for phase in ['train', 'val']:
            batchtime=time.time()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            optimizer.zero_grad()
            # Iterate over data.
            for id,(inputs, labels,cache_dicts) in enumerate(data_loaders[phase]):
                lo=len(data_loaders[phase])
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # zero the parameter gradients

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if id%1000==0:
                    print("process:[{} / {} ]".format(id,lo))
            if phase == 'train':
                torch.cuda.empty_cache()
                torch.save(model.state_dict(), 'alexnet_car_classier.pth')
            if phase == 'train':
                lr_scheduler.step()
            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            bathcend=time.time()-batchtime
            print('Using time: {:.4f}'.format(bathcend))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
        # 找出难负样本并添加到数据集中
        train_dataset = data_loaders['train'].dataset
        remain_negative_list = data_loaders['remain']
        jpeg_images = train_dataset.get_jpeg_images()
        transform = train_dataset.get_transform()
            #向前传播并添加难负样本到训练集中
        with torch.set_grad_enabled(False):
            #把剩余负样本构建成Dataset并放到DataLoader
            remain_dataset = CustomHardNegativeMiningDataset(
                remain_negative_list, jpeg_images, transform=transform)
            remain_data_loader = DataLoader(
                remain_dataset, batch_size=64, num_workers=8, drop_last=True)
            # 获取训练数据集的负样本集
            negative_list = train_dataset.get_negatives()
            # 记录后续增加的负样本
            add_negative_list = data_loaders.get('add_negative', [])
            running_corrects = 0
            # 遍历剩余数据集
            for inputs, labels, cache_dicts in remain_data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                #向前传播
                outputs = model(inputs)
                # print(outputs.shape)
                _, preds = torch.max(outputs, 1)
                #计算准确率
                running_corrects += torch.sum(preds == labels.data)
                #获取难负样本
                hard_negative_list, easy_neagtive_list = get_hard_negatives(preds.cpu().numpy(), cache_dicts)
                add_hard_negatives(hard_negative_list, negative_list, add_negative_list)
            #计算剩余样本的准确率
            remain_acc = running_corrects.double() / len(remain_negative_list)
            print('remiam negative size: {}, acc: {:.4f}'.format(len(remain_negative_list), remain_acc))
            # 找出了负样本后，添加到训练用的DataSet
            # 并更新DataLoader
            train_dataset.set_negative_list(negative_list)
            tmp_sampler = CustomBatchSampler(train_dataset.get_positive_num(),
                                             train_dataset.get_negative_num(),
                                             32, 96)
            data_loaders['train'] = DataLoader(train_dataset, batch_size=128, sampler=tmp_sampler,
                                               num_workers=8, drop_last=True)
            data_loaders['add_negative'] = add_negative_list
            # 重置数据集大小
            data_sizes['train'] = len(tmp_sampler)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    print("=====begin train SVM=====")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 载入数据
    data_loaders, data_sizes = load_data(r'./data/classifier')
    # 载入预训练好的模型
    model = models.alexnet()
    # print(model)
    # 修改最后一层为我们需要输出的特征，这里只需要检测2种，所以一种是背景一种是汽车
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2)
    # print(model)
    model.load_state_dict((torch.load('alexnet_car.pth')))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(num_features, 2)
    model = model.to(device)
    # 定义交叉损失
    criterion = hinge_loss
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # 共训练10轮，每隔4论减少一次学习率
    lr_schduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    # 训练模型和保存
    best_model = train_model(data_loaders, model, criterion,lr_schduler, optimizer, num_epochs=30, device=device)
    # 保存最好的模型参数
    torch.save(best_model.state_dict(), 'alexnet_car_classier.pth')
