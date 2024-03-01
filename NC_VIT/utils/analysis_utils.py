import torch
from torch.utils.data import DataLoader
from scipy.sparse.linalg import svds
import numpy as np
import torch.nn.functional as F

def compute_ece(probs, targets, n_bins=10):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = probs.argmax(dim=1) == targets
    confidences, predictions = torch.max(probs, 1)

    ece = torch.zeros(1, device=probs.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def analysis(graphs, model, args, criterion_summed, device, loader):
    class Features:
        pass
    def hook(self, input, output):
        Features.value = input[0].clone()
    def entropy(logits):
        p_softmax = F.softmax(logits, dim=1)
        return -(p_softmax * torch.log(p_softmax + 1e-5)).sum(dim=1)
    #classifier = model.head
    classifier = model.linear_head
    classifier.register_forward_hook(hook) 
    
    N               = [0 for _ in range(args.C)]
    mean            = [0 for _ in range(args.C)]
    Sw              = 0

    loss            = 0
    net_correct     = 0
    NCC_match_net   = 0
    total_features  = 0
    total_count     = 0
    batch_loss_correct = 0
    batch_loss_incorrect = 0
    batch_entropy_correct = []
    batch_entropy_incorrect = []
    Sw_cor, Sw_inc = 0, 0
    N_cor, N_inc = [0 for _ in range(args.C)], [0 for _ in range(args.C)]

    model.eval()

    for computation in ['Mean','Cov']:
        #pbar = tqdm(total=len(loader), position=0, leave=True)
        for batch_idx, (data, target) in enumerate(loader, start=1):

            data, target = data.to(device), target.to(device)
            output = model(data)
    
            if Features.value is not None:
                h = Features.value.data.view(data.shape[0],-1) # B CHW
                total_features += h.sum(dim=0)
                total_count += h.size(0)
            else:
                print("Warning: Hook has not been triggered!")
                continue

            preds = output.argmax(dim=1)
            correct = preds == target
            incorrect = ~correct

            if correct.any():
                loss_cor = criterion_summed(output[correct], target[correct]).item()
                entropy_cor = entropy(output[correct]).mean().item()
                batch_loss_correct += loss_cor
                batch_entropy_correct.append(entropy_cor)

            if incorrect.any():
                loss_inc = criterion_summed(output[incorrect], target[incorrect]).item()
                entropy_inc = entropy(output[incorrect]).mean().item()
                batch_loss_incorrect += loss_inc
                batch_entropy_incorrect.append(entropy_inc)


            # during calculation of class means, calculate loss
            if computation == 'Mean':
                if str(criterion_summed) == 'CrossEntropyLoss()':
                  loss += criterion_summed(output, target).item()
                elif str(criterion_summed) == "MultiMarginLoss()":
                    loss += criterion_summed(output, target).item()
                #elif str(criterion_summed) == 'MSELoss()':
                  #loss += criterion_summed(output, F.one_hot(target, num_classes=num_classes).float()).item()

            for c in range(args.C):
                # features belonging to class c
                idxs = (target == c).nonzero(as_tuple=True)[0]

                if len(idxs) == 0: # If no class-c in this batch
                  continue

                h_c = h[idxs,:] # B CHW

                if computation == 'Mean':
                    # update class means
                    mean[c] += torch.sum(h_c, dim=0) # CHW
                    N[c] += h_c.shape[0]

                elif computation == 'Cov':
                    # update within-class cov
                    z = h_c - mean[c].unsqueeze(0) # B CHW
                    cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                       z.unsqueeze(1))  # B 1 CHW
                    Sw += torch.sum(cov, dim=0)

                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    net_pred = torch.argmax(output[idxs,:], dim=1)
                    net_correct += sum(net_pred==target[idxs]).item()

                    # 2) agreement between prediction and nearest class center
                    NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                              for i in range(h_c.shape[0])])
                    NCC_pred = torch.argmin(NCC_scores, dim=1)
                    NCC_match_net += sum(NCC_pred==net_pred).item()

            if args.debug and batch_idx > 20:
                break
        #pbar.close()

        if computation == 'Mean':
            for c in range(args.C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)


    graphs.loss.append(loss)
    graphs.accuracy.append(net_correct/sum(N))
    graphs.NCC_mismatch.append(1-NCC_match_net/sum(N))
    #record correct and incorrect
    graphs.test_loss_correct.append(batch_loss_correct / sum(N))
    graphs.test_loss_incorrect.append(batch_loss_incorrect / sum(N))
    graphs.test_entropy_correct.append(sum(batch_entropy_correct) / (len(batch_entropy_correct)+1))
    graphs.test_entropy_incorrect.append(sum(batch_entropy_incorrect) / (len(batch_entropy_incorrect)+1))

    # loss with weight decay
    reg_loss = loss
    graphs.reg_loss.append(reg_loss)
    # global mean
    muG = torch.mean(M, dim=1, keepdim=True) # CHW 1

    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / args.C

    # avg norm
    avg_features = total_features / total_count
    avg_features =avg_features.sum(dim=0)
    W  = classifier.weight
    M_norms = torch.norm(M_,  dim=0)
    W_norms = torch.norm(W.T, dim=0)
    graphs.L2ofW.append(torch.sum(W_norms).item())
    graphs.avg_features.append(avg_features)
    graphs.L2ofH.append(torch.sum(M_norms).item())
    graphs.norm_M_CoV.append((torch.std(M_norms)/torch.mean(M_norms)).item())
    graphs.norm_W_CoV.append((torch.std(W_norms)/torch.mean(W_norms)).item())

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=args.C-1)
    inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T
    graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb))

    # ||W^T - M_||
    normalized_M = M_ / torch.norm(M_,'fro')
    normalized_W = W.T / torch.norm(W.T,'fro')
    graphs.W_M_dist.append((torch.norm(normalized_W - normalized_M)**2).item())

    # mutual coherence
    def coherence(V):
        G = V.T @ V
        G += torch.ones((args.C,args.C),device=device) / (args.C-1)
        G -= torch.diag(torch.diag(G))
        return torch.norm(G,1).item() / (args.C*(args.C-1))

    graphs.cos_M.append(coherence(M_/M_norms))
    graphs.cos_W.append(coherence(W.T/W_norms))

    return graphs

