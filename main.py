import argparse
import torch
import torch.nn as nn
import torchvision
import os

from torch.utils.data import random_split
from torchvision import transforms
from model import CNN, VGG, ResNet, mini_XCEPTION, DeepNN, AttentionCNN, AttentionDeepNN,AttentionResNet,AttentionVGG,Attentionmini_XCEPTION
from train import train_val_model
from test import test_model
from demo import demo

path_train = './data/train'
path_test = './data/test'



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms_train = transforms.Compose([
    transforms.Grayscale(),#使用ImageFolder默认扩展为三通道，重新变回去就行
    transforms.RandomHorizontalFlip(),#随机翻转
    transforms.ColorJitter(brightness=0.5, contrast=0.5),#随机调整亮度和对比度
    transforms.ToTensor()
])
transforms_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])
transforms_val = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

data_train = torchvision.datasets.ImageFolder(root=path_train,transform=transforms_train)
data_test = torchvision.datasets.ImageFolder(root=path_test,transform=transforms_test)
# 确定训练集和验证集的大小
total_size = len(data_train)
train_size = int(0.8 * total_size)  # 假设80%的数据用作训练集
val_size = total_size - train_size  # 剩余的20%用作验证集

# 随机分割数据集
data_train, data_val = random_split(data_train, [train_size, val_size])


def main():
    parser = argparse.ArgumentParser(description='Fer2013 Emotion Recognition')
    parser.add_argument('--model', type=str, default='CNN', help='Model to use: CNN, VGG, ResNet, mini_XCEPTION')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train, test, val, demo')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer: sgd, adam, rmsprop')
    parser.add_argument('--loss', type=str, default='cross_entropy', help='Loss function: cross_entropy, mse')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    path_model_param = 'checkpoint'
    # 确保保存目录存在
    if not os.path.exists(path_model_param):
        os.makedirs(path_model_param, exist_ok=True)

    # 构造文件名，包括模型类型和训练的epoch
    model_filename = f"{args.model}.pth"
    path_model_param = os.path.join(path_model_param, model_filename)

    # Initialize model
    model_dict = {
        'CNN': CNN(),
        'VGG': VGG(),
        'ResNet': ResNet(),
        'mini_XCEPTION': mini_XCEPTION(),
        'DeepNN': DeepNN(),
        'AttentionCNN': AttentionCNN(),
        'AttentionVGG': AttentionVGG(),
        'AttentionResNet': AttentionResNet(),
        'Attentionmini_XCEPTION': Attentionmini_XCEPTION(),
        'AttentionDeepNN': AttentionDeepNN()
    }
    model = model_dict[args.model].to(device)

    # Initialize optimizer and loss function
    optimizer_dict = {
        'sgd': torch.optim.SGD(model.parameters(), lr=args.lr),
        'adam': torch.optim.Adam(model.parameters(), lr=args.lr),
        'rmsprop': torch.optim.RMSprop(model.parameters(), lr=args.lr)
    }
    optimizer = optimizer_dict[args.optimizer]

    loss_dict = {
        'cross_entropy': nn.CrossEntropyLoss(),
        'mse': nn.MSELoss()
    }
    criterion = loss_dict[args.loss]

    # Data loaders

    train_loader = torch.utils.data.DataLoader(dataset=data_train,batch_size=args.batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=data_val,batch_size=args.batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=data_test,batch_size=args.batch_size,shuffle=False)

    if args.mode == 'train':
        train_val_model(model, train_loader, val_loader, optimizer, criterion, device = device, 
                    epochs=args.epochs, model_type = args.model, path_model_param = path_model_param)
    elif args.mode == 'test':
        test_model(model, test_loader, criterion, device = device, path_model_param = path_model_param)

    elif args.mode == 'demo':
        demo(model, device = device, path_model_param = path_model_param)

if __name__ == '__main__':
    main()