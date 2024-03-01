import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from loss.koleo_loss import KoLeoLoss
import math
import os
from PIL import ImageFilter
import random
import timm
def cosine_annealing_update(initial_lr, t, T_max, eta_min=0):
    """
    Calculate the learning rate for cosine annealing.

    Args:
    - initial_lr (float): The initial learning rate (i.e., eta_max).
    - t (int): The current epoch.
    - T_max (int): The maximum number of epochs.
    - eta_min (float, optional): The minimum learning rate. Default is 0.

    Returns:
    - float: The adjusted learning rate.
    """
    return eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(math.pi * t / T_max))

def smooth_labels(labels, num_classes, smoothing=0.1):
    """
    Returns tensor of shape [batch_size, num_classes]
    """
        
    soft_labels = torch.full((labels.size(0), num_classes), smoothing / (num_classes - 1), device=labels.device)
    soft_labels.scatter_(1, labels.data.unsqueeze(-1), 1.0 - smoothing)
    
    return soft_labels

def train(model, args, criterion, device, train_loader, optimizer, epoch):
    num_classes = args.C
    model.train()

    if args.pbar_show:
        pbar = tqdm(total=len(train_loader), position=0, leave=True)
    length = []
    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        if data.shape[0] != args.batch_size:
            continue

        data, target = data.to(device), target.to(device)
        soft_target = smooth_labels(target, num_classes=args.C, smoothing=1-args.confidence)  
        optimizer.zero_grad()
        out = model(data)


        if isinstance(criterion, KoLeoLoss):
            loss = criterion(out)
        elif isinstance(criterion, nn.CrossEntropyLoss):
            if args.labelsmoothing ==True:
                loss = -(soft_target * torch.log_softmax(out, dim=1)).sum(dim=1).mean()
            else:
                loss = criterion(out, target)
        elif isinstance(criterion,nn.MultiMarginLoss):
            loss = criterion(out, target)

        loss.backward()
        optimizer.step()
        
        accuracy = torch.mean((torch.argmax(out,dim=1)==target).float()).item()

        if args.pbar_show:
            pbar.update(1)
            pbar.set_description(
                'Train\t\tEpoch: {} [{}/{} ({:.0f}%)] \t'
                'Batch Loss: {:.6f} \t'
                'Batch Accuracy: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    len(train_loader),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                    accuracy))

        if args.debug and batch_idx > 2:
          break
    if args.pbar_show:
        pbar.close()

def set_optimizer(model, args):
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer == "AdamW":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def set_seed(random_seed):
    torch.manual_seed(random_seed)

def load_data(args):
    class GaussianBlur(object):
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

        def __init__(self, sigma=[0.1, 2.0]):
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x
    
    if args.dataset == "STL10":
        transform2 = transforms.Compose([transforms.Pad((args.padded_im_size - args.im_size)//2),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4467,0.4398,0.4066), (0.2603,0.2566, 0.2713))])
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4467,0.4398,0.4066), (0.2603,0.2566, 0.2713)),
        ])

        trans_valid = transforms.Compose(
        [transforms.Resize(224),  
         transforms.ToTensor(), 
         transforms.Normalize((0.4467,0.4398,0.4066), (0.2603,0.2566, 0.2713))])                             
        
        train_loader = torch.utils.data.DataLoader(
        datasets.STL10('data', split='train', download=True, transform=transform2),
        batch_size=args.batch_size, shuffle=True)

        analysis_loader = torch.utils.data.DataLoader(
        datasets.STL10('data', split='test', download=True, transform=transform2),
        batch_size=args.batch_size, shuffle=True)

    elif args.dataset == "CIFAR10":
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        trans_valid = transforms.Compose(
        [transforms.Resize(224),  
         transforms.ToTensor(), 
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) 

        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

        analysis_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, download=True, transform=trans_valid),
        batch_size=args.batch_size, shuffle=True, num_workers=1)


    elif args.dataset == "CIFAR100":
        transform2 = transforms.Compose([transforms.Pad((args.padded_im_size - args.im_size) // 2),
                                    transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615))])


        transform1 = transforms.Compose([transforms.RandomResizedCrop(36, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1,2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615))])

        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
        ])

        trans_valid = transforms.Compose(
        [transforms.Resize(224),  
         transforms.ToTensor(), 
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615))]) 

        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True)

        analysis_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=False, download=True, transform=transform_valid),
        batch_size=args.batch_size, shuffle=True)
        
    elif args.dataset == "tinyimagenet":
        transform1_tiny = transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1,2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        transform2 = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        ])
        data_folder = '/scratch/zz4330/data/'
        train_dataset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'train'), transform2)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        analysis_dataset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'val'), transform2)
        analysis_loader = torch.utils.data.DataLoader(analysis_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader, analysis_loader
