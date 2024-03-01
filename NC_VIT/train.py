import torch
import os
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.train_utils import train, set_optimizer, load_data, cosine_annealing_update
from utils.analysis_utils import analysis
from models.detached_resnet import Detached_ResNet
from tqdm import tqdm
from loss.koleo_loss import KoLeoLoss
import argparse
import pickle

class Graphs:
  def __init__(self):
    self.accuracy     = []
    self.loss         = []
    self.reg_loss     = []

    # NC1
    self.Sw_invSb     = []
    self.Sb           = []
    self.Sw_invSb     = []

    # NC2
    self.norm_M_CoV   = []
    self.norm_W_CoV   = []
    self.cos_M        = []
    self.cos_W        = []

    # NC3
    self.W_M_dist     = []

    # NC4
    self.NCC_mismatch = []

    # Decomposition
    self.MSE_wd_features = []
    self.LNC1 = []
    self.LNC23 = []
    self.Lperp = []

    #L2 Norm of W,M
    self.L2ofW = []
    self.L2ofH = []
    self.avg_features = []
    
    self.test_loss_correct = []
    self.test_loss_incorrect = []
    self.test_entropy_correct = []
    self.test_entropy_incorrect = []
    self.nc1_cor = []
    self.nc1_inc = []
class Graphs2:
  def __init__(self):
    self.accuracy     = []
    self.loss         = []
    self.reg_loss     = []

    # NC1
    self.Sw_invSb     = []
    self.Sb           = []
    self.Sw_invSb     = []

    # NC2
    self.norm_M_CoV   = []
    self.norm_W_CoV   = []
    self.cos_M        = []
    self.cos_W        = []

    # NC3
    self.W_M_dist     = []

    # NC4
    self.NCC_mismatch = []

    # Decomposition
    self.MSE_wd_features = []
    self.LNC1 = []
    self.LNC23 = []
    self.Lperp = []

    #L2 Norm of W
    self.L2ofW = []
    self.L2ofH = []
    self.avg_features = []

    self.test_loss_correct = []
    self.test_loss_incorrect = []
    self.test_entropy_correct = []
    self.test_entropy_incorrect = []
    self.nc1_cor = []
    self.nc1_inc = []

def train_model(model, device, args): 
    train_loader, analysis_loader = load_data(args)
    optimizer = set_optimizer(model, args)

    graphs, graphs2 = Graphs(), Graphs2()

    if args.loss_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
        criterion_summed = nn.CrossEntropyLoss(reduction='sum')
    elif args.loss_name == 'KoLeoLoss': 
        criterion = KoLeoLoss(weight=0.5)
        criterion_summed = KoLeoLoss(reduction='sum',weight=0.5)
    elif args.loss_name ==  'MultiMarginLoss':
        criterion = nn.MultiMarginLoss()
        criterion_summed = nn.MultiMarginLoss(reduction='sum')
    initial_lr = args.lr    
    if not args.lrdecay_cos:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=args.epochs_lr_decay,
                                                      gamma=args.lr_decay)

    cur_epochs = []
    for epoch in range(1, args.epochs + 1):
        train(model, args, criterion, device, train_loader, optimizer, epoch)
        if args.lrdecay_cos:
            adjusted_lr = cosine_annealing_update(initial_lr, epoch, 400)
            for param_group in optimizer.param_groups:
                param_group['lr'] = adjusted_lr
        else:
            lr_scheduler.step()

        if epoch in args.epoch_list:
            cur_epochs.append(epoch)
            graphs = analysis(graphs, model, args, criterion_summed, device, analysis_loader)
            graphs2 = analysis(graphs2, model, args, criterion_summed, device, train_loader)
            
            plot_graphs(epoch, args.fig_saving_pth, cur_epochs, graphs, graphs2)

def plot_graphs(epoch, fig_saving_pth, cur_epochs, graphs, graphs2):
    os.makedirs(fig_saving_pth, exist_ok=True)

    plt.gcf().subplots_adjust(bottom=0.15)
    infile=open(f"{fig_saving_pth}record.txt",'w')
    plt.figure(1)
    plt.semilogy(cur_epochs, graphs.reg_loss)
    plt.semilogy(cur_epochs, graphs2.reg_loss)
    print("loss_test:")
    print(graphs.reg_loss)
    print("loss_train:")
    print(graphs2.reg_loss)
    plt.legend(['Loss of Test','Loss of Train'])
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training and Test Loss')
    plt.savefig(fig_saving_pth + "1loss.png")

    plt.figure(2)
    plt.plot(cur_epochs, 100*(1 - np.array(graphs.accuracy)))
    plt.plot(cur_epochs, 100*(1 - np.array(graphs2.accuracy)))
    print("error_test:")
    print(100*(1 - np.array(graphs.accuracy)))
    print("error_train:")
    print(100*(1 - np.array(graphs2.accuracy)))
    plt.legend(['Error of Test','Error of Train'])
    plt.xlabel('Epoch')
    plt.ylabel('Training and Test Error (%)')
    plt.title('Training and Test Error')
    plt.savefig(fig_saving_pth + "error.png")
    
    plt.figure(3)
    plt.semilogy(cur_epochs, graphs.Sw_invSb)
    plt.semilogy(cur_epochs, graphs2.Sw_invSb)
    print("nc1_test:")
    print(graphs.Sw_invSb)
    print("nc1_train:")
    print(graphs2.Sw_invSb)
    plt.xlabel('Epoch')
    plt.ylabel('Tr{Sw Sb^-1}')
    plt.legend(['Test','Train'])
    plt.title('NC1: Activation Collapse')
    plt.savefig(fig_saving_pth + "nc1.png")

    plt.figure(4)
    plt.plot(cur_epochs, graphs.norm_M_CoV)
    plt.plot(cur_epochs, graphs2.norm_M_CoV)
    plt.plot(cur_epochs, graphs.norm_W_CoV)
    print("nc2-1_test:")
    print(graphs.norm_M_CoV)
    print("nc2-1_train:")
    print(graphs2.norm_M_CoV)
    plt.legend(['Class Means of Test','Class Means of Train','Classifier'])
    plt.xlabel('Epoch')
    plt.ylabel('Std/Avg of Norms')
    plt.title('NC2: Equinorm')
    plt.savefig(fig_saving_pth + "nc2.1.png")
    
    plt.figure(5)
    plt.plot(cur_epochs, graphs.cos_M)
    plt.plot(cur_epochs, graphs2.cos_M)
    plt.plot(cur_epochs, graphs.cos_W)
    print("nc2-2_test:")
    print(graphs.cos_M)
    print("nc2-2_train:")
    print(graphs2.cos_M)
    plt.legend(['Class Means of Test','Class Means of Train','Classifier'])
    plt.xlabel('Epoch')
    plt.ylabel('Avg|Cos + 1/(C-1)|')
    plt.title('NC2: Maximal Equiangularity')
    plt.savefig(fig_saving_pth + "nc2.2.png")

    plt.figure(6)
    plt.plot(cur_epochs,graphs.W_M_dist)
    plt.plot(cur_epochs,graphs2.W_M_dist)
    print("nc3_test:")
    print(graphs.W_M_dist)
    print("nc3_train:")
    print(graphs2.W_M_dist)
    plt.xlabel('Epoch')
    plt.ylabel('||W^T - H||^2')
    plt.legend(['Test','Train'])
    plt.title('NC3: Self Duality')
    plt.savefig(fig_saving_pth + "nc3.png")

    plt.figure(7)
    plt.plot(cur_epochs,graphs.NCC_mismatch)
    plt.plot(cur_epochs,graphs2.NCC_mismatch)
    print("nc4_test:")
    print(graphs.NCC_mismatch)
    print("nc4_train:")
    print(graphs2.NCC_mismatch)
    plt.xlabel('Epoch')
    plt.ylabel('Proportion Mismatch from NCC')
    plt.legend(['Test','Train'])
    plt.title('NC4: Convergence to NCC')
    plt.savefig(fig_saving_pth + "nc4.png")

    plt.figure(8)
    plt.plot(cur_epochs,graphs.L2ofW)
    plt.plot(cur_epochs,graphs2.L2ofW)
    print("Wnorm_test:")
    print(graphs.L2ofW)
    print("Wnorm_train:")
    print(graphs2.L2ofW)
    plt.xlabel('Epoch')
    plt.ylabel('Norm of Classifier')
    plt.legend(['Test','Train'])
    plt.title('W_norm')
    plt.savefig(fig_saving_pth + "W.png")

    plt.figure(9)
    plt.plot(cur_epochs,graphs.L2ofH)
    plt.plot(cur_epochs,graphs2.L2ofH)
    print("Wnorm_test:")
    print(graphs.L2ofH)
    print("Wnorm_train:")
    print(graphs2.L2ofH)
    plt.xlabel('Epoch')
    plt.ylabel('Norm of Feature Extractor')
    plt.legend(['Test','Train'])
    plt.title('H_norm')
    plt.savefig(fig_saving_pth + "H.png")

    plt.figure(figsize=(10, 6))
    plt.plot(cur_epochs, graphs.loss, label='Loss')
    plt.plot(cur_epochs, graphs.test_loss_correct, label='Test loss correct')
    plt.plot(cur_epochs, graphs.test_loss_incorrect, label='Test loss incorrect')
    plt.plot(cur_epochs, graphs.test_entropy_correct, label='Test entropy correct', linestyle='--')
    plt.plot(cur_epochs, graphs.test_entropy_incorrect, label='Test entropy incorrect', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Test loss/Test entropy')
    plt.title('Model Evaluation Metrics Over Epochs')
    plt.legend()
    plt.savefig(fig_saving_pth + "metrics_plot.png")
    plt.close()
    if epoch % 10 == 0:
        with open(f"{fig_saving_pth}graphs_{epoch}.pkl", 'wb') as file:
            pickle.dump(graphs, file)
        with open(f"{fig_saving_pth}graphs2_{epoch}.pkl", 'wb') as file:
            pickle.dump(graphs2, file)
        with open(os.path.join(fig_saving_pth, "metrics_record.txt"), 'w') as file:
            file.write(f"At epoch {epoch}:\n")
            file.write(f"Loss: {graphs.loss}\n")
            file.write(f"Test loss correct: {graphs.test_loss_correct}\n")
            file.write(f"Test loss incorrect: {graphs.test_loss_incorrect}\n")
            file.write(f"Test entropy correct: {graphs.test_entropy_correct}\n")
            file.write(f"Test entropy incorrect: {graphs.test_entropy_incorrect}\n")
            file.write(f"At {epoch} epoch:")
            file.write(f"Loss of Test: {graphs.reg_loss} \n")
            file.write(f"Loss of Train: {graphs2.reg_loss} \n ")
            file.write(f"Error of Test: {100*(1 - np.array(graphs.accuracy))} \n")
            file.write(f"Error of Train: {100*(1 - np.array(graphs2.accuracy))} \n")
            file.write(f"NC1 of Test: {graphs.Sw_invSb}\n")
            file.write(f"NC1 of Train: {graphs2.Sw_invSb}\n")
            file.write(f"NC2.1 of Test: {graphs.norm_M_CoV}\n")
            file.write(f"NC2.1 of Train: {graphs2.norm_M_CoV}\n")
            file.write(f"NC2.1 of W: {graphs.norm_W_CoV}\n")
            file.write(f"NC2.2 of Test: {graphs.cos_M}\n")
            file.write(f"NC2.2 of Train: {graphs2.cos_M}\n")
            file.write(f"NC2.2 of W: {graphs2.cos_W}\n")
            file.write(f"NC3 of Test: {graphs.W_M_dist}\n")
            file.write(f"NC3 of Train: {graphs2.W_M_dist}\n")
            file.write(f"NC4 of Test: {graphs.NCC_mismatch}\n")
            file.write(f"NC4 of Train: {graphs2.NCC_mismatch}\n")
            file.write(f"W_norm of Test: {graphs.L2ofW}\n")
            file.write(f"W_norm of Train: {graphs2.L2ofW}\n")
            file.write(f"H_norm of Test: {graphs.L2ofH}\n")
            file.write(f"H_norm of Train: {graphs2.L2ofH}\n")
        print("L2 of W:" + str(graphs.L2ofW))
        print("Avg_f:" + str(graphs.avg_features))
    if epoch == 800 or epoch == 350 :
        final_label=len(graphs.reg_loss)-1
        plt.figure(1)
        plt.semilogy(cur_epochs, graphs.reg_loss)
        plt.text(cur_epochs[final_label], graphs.reg_loss[final_label],str(graphs.reg_loss[final_label]))
        plt.semilogy(cur_epochs, graphs2.reg_loss)
        plt.text(cur_epochs[final_label], graphs2.reg_loss[final_label],str(graphs2.reg_loss[final_label]))
        plt.legend(['Loss of Test','Loss of Train'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training and Test Loss')
        plt.savefig(fig_saving_pth + f"loss at {epoch}.png")

        plt.figure(2)
        plt.plot(cur_epochs, 100*(1 - np.array(graphs.accuracy)))
        plt.text(cur_epochs[final_label], 100*(1 - np.array(graphs.accuracy[final_label])),str(100*(1 - np.array(graphs.accuracy[final_label]))))
        plt.plot(cur_epochs, 100*(1 - np.array(graphs2.accuracy)))
        plt.text(cur_epochs[final_label], 100*(1 - np.array(graphs2.accuracy[final_label])),str(100*(1 - np.array(graphs2.accuracy[final_label]))))
        print("error_test:")
        print(100*(1 - np.array(graphs.accuracy)))
        print("error_train:")
        print(100*(1 - np.array(graphs2.accuracy)))
        plt.legend(['Error of Test','Error of Train'])
        plt.xlabel('Epoch')
        plt.ylabel('Training and Test Error (%)')
        plt.title('Training and Test Error')
        plt.savefig(fig_saving_pth + f"error at {epoch}.png")

        plt.figure(3)
        plt.semilogy(cur_epochs, graphs.Sw_invSb)
        plt.text(cur_epochs[final_label],graphs.Sw_invSb[final_label],str(graphs.Sw_invSb[final_label]))
        plt.semilogy(cur_epochs, graphs2.Sw_invSb)
        plt.text(cur_epochs[final_label],graphs2.Sw_invSb[final_label],str(graphs2.Sw_invSb[final_label]))
        print("nc1_test:")
        print(graphs.Sw_invSb)
        print("nc1_train:")
        print(graphs2.Sw_invSb)
        plt.xlabel('Epoch')
        plt.ylabel('Tr{Sw Sb^-1}')
        plt.legend(['Test','Train'])
        plt.title('NC1: Activation Collapse')
        plt.savefig(fig_saving_pth + f"nc1 at {epoch}.png")

        plt.figure(4)
        plt.plot(cur_epochs, graphs.norm_M_CoV)
        plt.text(cur_epochs[final_label],graphs.norm_M_CoV[final_label],str(graphs.norm_M_CoV[final_label]))
        plt.plot(cur_epochs, graphs2.norm_M_CoV)
        plt.text(cur_epochs[final_label],graphs2.norm_M_CoV[final_label],str(graphs2.norm_M_CoV[final_label]))
        plt.plot(cur_epochs, graphs.norm_W_CoV)
        plt.text(cur_epochs[final_label],graphs.norm_W_CoV[final_label],str(graphs.norm_W_CoV[final_label]))
        print("nc2-1_test:")
        print(graphs.norm_M_CoV)
        print("nc2-1_train:")
        print(graphs2.norm_M_CoV)
        plt.legend(['Class Means of Test','Class Means of Train','Classifier'])
        plt.xlabel('Epoch')
        plt.ylabel('Std/Avg of Norms')
        plt.title('NC2: Equinorm')
        plt.savefig(fig_saving_pth + f"nc2.1 at {epoch}.png")

        plt.figure(5)
        plt.plot(cur_epochs, graphs.cos_M)
        plt.text(cur_epochs[final_label],graphs.cos_M[final_label],str(graphs.cos_M[final_label]))
        plt.plot(cur_epochs, graphs2.cos_M)
        plt.text(cur_epochs[final_label],graphs2.cos_M[final_label],str(graphs2.cos_M[final_label]))
        plt.plot(cur_epochs, graphs.cos_W)
        plt.text(cur_epochs[final_label],graphs.cos_W[final_label],str(graphs.cos_W[final_label]))
        print("nc2-2_test:")
        print(graphs.cos_M)
        print("nc2-2_train:")
        print(graphs2.cos_M)
        plt.legend(['Class Means of Test','Class Means of Train','Classifier'])
        plt.xlabel('Epoch')
        plt.ylabel('Avg|Cos + 1/(C-1)|')
        plt.title('NC2: Maximal Equiangularity')
        plt.savefig(fig_saving_pth + f"nc2.2 at {epoch}.png")

        plt.figure(6)
        plt.plot(cur_epochs,graphs.W_M_dist)
        plt.text(cur_epochs[final_label],graphs.W_M_dist[final_label],str(graphs.W_M_dist[final_label]))
        plt.plot(cur_epochs,graphs2.W_M_dist)
        plt.text(cur_epochs[final_label],graphs2.W_M_dist[final_label],str(graphs2.W_M_dist[final_label]))
        print("nc3_test:")
        print(graphs.W_M_dist)
        print("nc3_train:")
        print(graphs2.W_M_dist)
        plt.xlabel('Epoch')
        plt.ylabel('||W^T - H||^2')
        plt.legend(['Test','Train'])
        plt.title('NC3: Self Duality')
        plt.savefig(fig_saving_pth + f"nc3 at {epoch}.png")

        plt.figure(7)
        plt.plot(cur_epochs,graphs.NCC_mismatch)
        plt.text(cur_epochs[final_label],graphs.NCC_mismatch[final_label],str(graphs.NCC_mismatch[final_label]))
        plt.plot(cur_epochs,graphs2.NCC_mismatch)
        plt.text(cur_epochs[final_label],graphs2.NCC_mismatch[final_label],str(graphs2.NCC_mismatch[final_label]))
        print("nc4_test:")
        print(graphs.NCC_mismatch)
        print("nc4_train:")
        print(graphs2.NCC_mismatch)
        plt.xlabel('Epoch')
        plt.ylabel('Proportion Mismatch from NCC')
        plt.legend(['Test','Train'])
        plt.title('NC4: Convergence to NCC')
        plt.savefig(fig_saving_pth + f"nc4 at {epoch}.png")

        plt.figure(8)

        plt.plot(cur_epochs,graphs.L2ofW)
        plt.text(cur_epochs[final_label],graphs.L2ofW[final_label],str(graphs.L2ofW[final_label]))
        plt.plot(cur_epochs,graphs2.L2ofW)
        plt.text(cur_epochs[final_label],graphs2.L2ofW[final_label],str(graphs2.L2ofW[final_label]))
        print("Wnorm_test:")
        print(graphs.L2ofW)
        print("Wnorm_train:")
        print(graphs2.L2ofW)
        plt.xlabel('Epoch')
        plt.ylabel('Norm of Classifier')
        plt.legend(['Test','Train'])
        plt.title('W_norm')
        plt.savefig(fig_saving_pth + f"W at {epoch}.png")

        plt.figure(9)
        plt.plot(cur_epochs,graphs.L2ofH)
        plt.text(cur_epochs[final_label],graphs.L2ofH[final_label],str(graphs.L2ofH[final_label]))
        plt.plot(cur_epochs,graphs2.L2ofH)
        plt.text(cur_epochs[final_label],graphs2.L2ofH[final_label],str(graphs2.L2ofH[final_label]))
        print("Hnorm_test:")
        print(graphs.L2ofH)
        print("Hnorm_train:")
        print(graphs2.L2ofH)
        plt.xlabel('Epoch')
        plt.ylabel('Norm of Feature Extrator')
        plt.legend(['Test','Train'])
        plt.title('H_norm')
        plt.savefig(fig_saving_pth + f"H at {epoch}.png")
    plt.show()
    infile.close()