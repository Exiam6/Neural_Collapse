import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Change_WD_V2')
    # General parameters
    parser.add_argument('--fig_saving_pth', type=str, default='/scratch/zz4330/NC_VIT/Result/CIFAR10_vit_tiny_new/', help='Figure saving path')
    parser.add_argument('--feature_decay', type=float, default=5e-4, help='Feature decay strength')
    parser.add_argument('--classifier_decay', type=float, default=5e-4, help='Classifier decay strength')

    parser.add_argument('--debug', default=False, help='Only runs 20 batches per epoch for debugging, and set debug to true', action='store_true')
    parser.add_argument('--pbar_show', default=True, help='Close pbar',action='store_false')
    parser.add_argument('--random_seed', type=int, default=43, help='Random seed')
    parser.add_argument('--epochs', type=int, default=801, help='Number of epochs')
    parser.add_argument('--epochs_lr_decay', nargs='+', type=int, default=[200,400], help='Epochs for learning rate decay')
    #parser.add_argument('--epochs_lr_decay', nargs='+', type=int, default=[parser.parse_args().epochs * i // 7 for i in range(1, 7)], help='Epochs for learning rate decay')
    parser.add_argument('--epoch_list', nargs='+', type=int, default=[i*10 for i in range(parser.parse_args().epochs//10+1)], help = 'List of epochs for analysis')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str,  default='CIFAR10', choices=['CIFAR10', 'MINIST','STL10','tinyimagenet','CIFAR100'], help='Name of the dataset')
    parser.add_argument('--im_size', type=int, default=32, help='Image size')
    parser.add_argument('--padded_im_size', type=int, default=36, help='Padded image size')
    parser.add_argument('--C', type=int, default=100, help='Number of classes')
    parser.add_argument('--input_ch', type=int, default=3, help='Number of input channels')

    # Optimization parameters
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD','AdamW'], help='Optimizer')
    parser.add_argument('--loss_name', type=str, default='CrossEntropyLoss', choices=['KoLeoLoss','CrossEntropyLoss', 'MSELoss','MultiMarginLoss'], help='Loss function')
    parser.add_argument('--lr', type=float, default=1e-4 , help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.25, help='Learning rate decay factor')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum value')

    # Regularization parameters
    
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay strength')
    parser.add_argument('--length_regularization_strength', type=float, default=1e-5, help='Length regularization strength')
    parser.add_argument('--lrdecay_cos',default=True)
    parser.add_argument('--labelsmoothing',default=False)
    parser.add_argument('--confidence', type=float, default=0.90)
    parser.add_argument('--split',type=int,default=50000)
    return parser.parse_args()

def set_learning_rate(args):
    if args.loss_name == 'CrossEntropyLoss':
        args.lr = 0.0679
    elif args.loss_name == 'MSELoss':
        args.lr = 0.0184
    return args

def set_dataset_para(args):
    if args.dataset == 'CIFAR10':
        args.im_size        = 32
        args.padded_im_size = 36
        args.C              = 10
        args.input_ch       = 3
    elif args.dataset == 'MINIST':
        args.im_size        = 28
        args.padded_im_size = 32
        args.C              = 10
        args.input_ch       = 1
    elif args.dataset == 'STL10':
        args.im_size        = 96
        args.padded_im_size = 100
        args.C              = 10
        args.input_ch       = 3
    elif args.dataset == "tinyimagenet":
        args.im_size        = 64
        args.padded_im_size = 68
        args.C              = 200
        args.input_ch       = 3
    elif args.dataset == 'CIFAR100':
        args.im_size        = 32
        args.padded_im_size = 36
        args.C              = 100
        args.input_ch       = 3
    return args
