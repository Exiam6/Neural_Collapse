import torch
import argparse
from torchvision.transforms import GaussianBlur, RandomApply
from utils.train_utils import set_seed
from models.detached_resnet import Detached_ResNet
from train import train_model
from configs import configs
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from models.vit_small import ViT

def get_vit_model():
    v = ViT(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=256,
        depth=4,
        heads=6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    return v

def main():
    # Parsing arguments 
    args = configs.get_args()
    args = configs.set_learning_rate(args)
    args = configs.set_dataset_para(args)

    # Set random seed
    set_seed(args.random_seed)

    # Initialize the device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    #num_classes = 10
    #model.head = nn.Linear(model.head.in_features, num_classes)
    model = get_vit_model()
    model = model.to(device)
    
    train_model(model, device, args)

if __name__ == '__main__':
    main()