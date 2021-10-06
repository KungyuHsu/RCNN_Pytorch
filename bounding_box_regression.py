
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import AlexNet
from matplotlib import pyplot as plt
from data.bbox_regression import BBoxRegressionDataset
import util
import numpy as np


def load_data(data_root_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    data_set = BBoxRegressionDataset(data_root_dir, transform=transform)
    data_loader = DataLoader(data_set, batch_size=128, shuffle=True, num_workers=4)

    return data_loader


def train_model(data_loader,test_loader, feature_model, model, criterion, optimizer, lr_scheduler, num_epochs=25,device=None):
    since = time.time()

    model.train()  # Set model to training mode
    loss_list = list()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        valrunning_loss=0.0
        vallossslist=[]
        # Iterate over data.
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.float().to(device)

            features = feature_model.features(inputs)
            features = torch.flatten(features, 1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(features)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            lr_scheduler.step()



        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            with torch.no_grad():
                features = feature_model.features(inputs)
                features = torch.flatten(features, 1)
                # forward
                outputs = model(features)
                valloss = criterion(outputs, targets)

                # statistics
                valrunning_loss += valloss.item() * inputs.size(0)
        val_loss = valrunning_loss / test_loader.dataset.__len__()
        epoch_loss = running_loss / data_loader.dataset.__len__()
        loss_list.append(epoch_loss)
        vallossslist.append(val_loss)


        if epoch>1:
            np.savetxt("./valloss.npy",vallossslist, fmt='%d', delimiter=' ')
            np.savetxt("./epochloss.npy",loss_list, fmt='%d', delimiter=' ')
        print('epoch Loss: {:.4f}'.format(epoch_loss))
        print('val Loss: {:.4f}'.format(val_loss))
        print('pth: {:.4f}'.format(epoch%10))
        plt.plot(loss_list)
        plt.savefig("epoch.png")
        plt.plot(vallossslist)
        plt.savefig("val.png")
        plt.clf()
        # 每训练十轮就保存
        torch.save(model, './models/bbox_regression_%d.pth' % (epoch%10))

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.save(model, './models/bbox_regression.pth')
    return loss_list


def get_model(device=None):
    # 加载CNN模型
    model = AlexNet(num_classes=2)
    model.load_state_dict(torch.load('alexnet_car_classier.pth'))
    model.eval()

    # 取消梯度追踪
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model


if __name__ == '__main__':
    data_loader = load_data('./data/bbr/train')
    test_loader = load_data('./data/bbr/val')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_model = get_model(device)

    # AlexNet最后一个池化层计算得到256*6*6输出
    in_features = 256 * 6 * 6
    out_features = 4
    # model = nn.Linear(in_features, out_features)
    model = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.Linear(512, 128),
        nn.Linear(128, 54),
        nn.Linear(54,out_features)
    )
    model.to(device)

    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    loss_list = train_model(data_loader, test_loader, feature_model, model, criterion, optimizer, lr_scheduler, device=device,
                            num_epochs=3000 )
    plt.plot(loss_list)
    plt.savefig("./res.png")
    plt.show()