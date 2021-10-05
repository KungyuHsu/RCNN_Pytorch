
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from data.finetune_dataset import CustomFinetuneDataset
from data.finetune_sample import CustomBatchSampler

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
        #定义Dataset
        data_set = CustomFinetuneDataset(data_dir, transform=transform)
        #定义sampler
        data_sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(), 32, 96)
        #定义加载器
        data_loader = DataLoader(data_set, batch_size=128, sampler=data_sampler, num_workers=8, drop_last=True)

        data_loaders[name] = data_loader
        data_sizes[name] = data_sampler.__len__()

    return data_loaders, data_sizes


def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        torch.cuda.empty_cache()
        # Each epoch has a training and validation phase
        ammu=4
        for phase in ['train', 'val']:
            batchtime=time.time()
            #batchNormal
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            optimizer.zero_grad()
            # Iterate over data.
            for id,(inputs, labels) in enumerate(data_loaders[phase]):
                lo=len(data_loaders[phase])
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss = loss/ ammu
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if (id+1)/ammu==0:
                            optimizer.step()
                            optimizer.zero_grad()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if id%10==0:
                    print("process:[{} / {} ".format(id,lo))
            if phase == 'train' and (id+1)/ammu==0:
                torch.cuda.empty_cache()
                lr_scheduler.step()
                torch.save(model.state_dict(), 'alexnet_car.pth')
            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]
            torch.cuda.empty_cache()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            bathcend=time.time()-batchtime
            print('Using time: {:.4f}'.format(bathcend))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 载入数据
    data_loaders, data_sizes = load_data(r'.\data\funetune')
    # 载入预训练好的模型
    model = models.alexnet(pretrained=True)
    # 修改最后一层为我们需要输出的特征，这里只需要检测2种，所以一种是背景一种是汽车
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2)
    # 传送到GPU
    model = model.to(device)
    # 定义交叉损失
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # 动态学习率，不过原论文直接定义1e-3
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # 训练模型和保存
    best_model = train_model(data_loaders, model, criterion, optimizer, lr_scheduler, device=device, num_epochs=25)
    # 保存最好的模型参数
    torch.save(best_model.state_dict(), 'alexnet_car.pth')
