import os
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from network import SJNET
from validation import validate
from dataset import Kinetics_GEBD_train, Kinetics_GEBD_validation, Kinetics_GEBD_test
from tqdm import tqdm
from config import *
from torch.distributions import Categorical
from torch.multiprocessing import set_start_method
#from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda')

import warnings
warnings.filterwarnings("ignore")

def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def get_max_key_value(dictionary: dict):
    keys = dictionary.keys()
    max_value = -1
    max_key = -1
    for key in keys:
        if dictionary[key] > max_value:
            max_key = key
            max_value = dictionary[key]
    return max_key, max_value

def get_basic_mask(gap=5):
    tmp = [[i,j] for i in range(FEATURE_LEN) for j in range(FEATURE_LEN) if np.abs(i-j) <= gap]
    x, y = zip(*tmp)
    basic_mask = np.zeros((FEATURE_LEN, FEATURE_LEN))
    basic_mask[x, y]= 1
    basic_mask = torch.from_numpy(basic_mask.astype(bool))
    return basic_mask

def get_mask(tsm, annotations):
    indices = torch.nonzero(annotations).cpu().numpy()
    positive_mask = torch.eye(tsm.size(2)).unsqueeze(0).repeat(tsm.size(0), 1, 1)
    neutral_mask = torch.zeros(tsm.size(0), tsm.size(2), tsm.size(3))
    current_batch = 0
    current_start = 0
    for i,j in indices:
        if i != current_batch:
            positive_mask[current_batch, current_start:, current_start:] = 1
            current_batch = i
            current_start = 0
        positive_mask[i, current_start:j, current_start:j] = 1
        neutral_mask[i, j, :] = 1
        neutral_mask[i, :, j] = 1
        neutral_mask[i, j, j] = 0
        current_start = j+1
    return positive_mask.bool().unsqueeze(1), neutral_mask.bool().unsqueeze(1)

def get_length(feature):
    #IN: feature: [40, 4096]
    last = feature[-1]
    for i in reversed(range(len(feature)-1)):
        if not torch.equal(feature[i], last):
            ret = i+1
            break
    return ret

def g(unit_square):
    const = UNIT_SQUARE_SIZE//2
    p1 = unit_square[:const, :const]
    p2 = unit_square[const+1:, const+1:]
    n1 = unit_square[:const, const+1:]
    n2 = unit_square[const+1:, :const]
    score = torch.mean((p1+p2)-(n1+n2))
    return score

def f(tsm, start_index):
    #tsm: [l, l]
    original_len = tsm.size(0)
    #print(f'input: {tsm.size()}')
    if original_len < MINIMUN_SQUARE_SIZE:
        return []
    topk = int(original_len*TOPK_RATIO)
    score_list = []
    pad_tsm = nn.ZeroPad2d(UNIT_SQUARE_SIZE//2)(tsm)
    for i in range(UNIT_SQUARE_SIZE//2, original_len+(UNIT_SQUARE_SIZE//2)):
        unit_square = pad_tsm[i-(UNIT_SQUARE_SIZE//2):i+(UNIT_SQUARE_SIZE//2)+1, i-(UNIT_SQUARE_SIZE//2):i+(UNIT_SQUARE_SIZE//2)+1]
        score_list.append(g(unit_square))
    scores = torch.stack(score_list)
    if torch.max(scores) - torch.mean(scores) < MIN_DIFF:
        return []
    threshold = -torch.kthvalue(-scores, topk).values
    modified_scores = torch.where(scores >= threshold, scores, torch.full_like(scores, BIG_MINUS))
    distribution = Categorical(logits=modified_scores)
    index = distribution.sample().cpu().numpy().item()
    pre_indice = f(tsm[:index,:index], start_index)
    post_indice = f(tsm[index+1:,index+1:], start_index+index+1)
    ret = pre_indice + post_indice
    ret.append(start_index + index)
    return ret

if __name__ == '__main__':
    if SAVE_MODEL:
        model_path = os.path.join(MODEL_SAVE_PATH, VER)
        os.makedirs(model_path, exist_ok=True)
    if SAVE_RESULT:
        result_path = os.path.join(RESULT_PATH, VER)
        os.makedirs(result_path, exist_ok=True)

    #writer = SummaryWriter(os.path.join("logs", VER))

    try:
        set_start_method('spawn')
    except RuntimeError as e:
        print(e)
        pass

    torch.set_printoptions(threshold=np.inf, sci_mode=False)
    torch.autograd.set_detect_anomaly(True)
    criterion = nn.BCEWithLogitsLoss()
    bceloss = nn.BCELoss()
    basic_mask = get_basic_mask(gap=GAP)
    loss_list = []

    network = nn.DataParallel(SJNET()).to(device)

    print(f"# of trainable parameters of model '{VER}': {count_parameters(network, only_trainable=True)}")
    print(f"# of all parameters of model '{VER}': {count_parameters(network)}")

    # for early stopping
    no_improvement_duration = 0
    improve_flag = True
    best_f1 = 0

    train_dataloader = DataLoader(Kinetics_GEBD_train(), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    validation_dataloader = DataLoader(Kinetics_GEBD_validation(), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(Kinetics_GEBD_test(), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    for epoch in range(EPOCHS):
        # check early stopping
        if not improve_flag:
            no_improvement_duration += 1
            if no_improvement_duration >= PATIENCE:
                print("EARLY STOPPING !!")
                break
        improve_flag = False

        epoch_loss_list = []
        network.train()

        print(f"TRAIN - EPOCH {epoch}")
        for i, (feature, _, _, _) in enumerate(tqdm(train_dataloader)):
            feature = feature.to(device)
            tsm = network(feature)
            whole_boundaries = torch.zeros(len(tsm), FEATURE_LEN).to(device)
            mean_tsm = torch.mean(tsm, dim=1)
            for i, x in enumerate(mean_tsm):
                length = get_length(feature[i])
                boundary = sorted(f(x, 0))
                boundary = [i for i in boundary if i < length]
                whole_boundaries[i][boundary] = 1
            _basic_mask = basic_mask.repeat(tsm.size(0), tsm.size(1),1,1)
            #tsm loss
            whole_mask, neutral_mask = get_mask(tsm, whole_boundaries)
            whole_mask = whole_mask.repeat(1,CHANNEL_NUM,1,1)
            whole_anti_mask = torch.logical_not(whole_mask)
            whole_mask = torch.logical_and(_basic_mask, whole_mask)
            whole_anti_mask = torch.logical_and(torch.logical_and(_basic_mask, whole_anti_mask), torch.logical_not(neutral_mask))
            whole_aux_loss = torch.mean(tsm[whole_anti_mask]) - torch.mean(tsm[whole_mask])
            
            loss = whole_aux_loss
            network.module.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=2.0)
            network.module.opt.step()
            epoch_loss_list.append(loss.detach().cpu().numpy())
        
        network.eval()
        fig = plt.figure(figsize=(8,8))
        fig.add_subplot(2,3,1)
        if epoch != 0:
            loss_list.append(sum(epoch_loss_list)/len(epoch_loss_list))
            plt.plot(list(range(len(loss_list))), loss_list)
        
        vis_batch = []               
        for vid_name in VAL_VIDEOS:
            path = f'{VISUAL_DATA_PATH}{vid_name}'
            vis_batch.append(np.load(path))
        vis_batch = torch.from_numpy(np.stack(vis_batch, axis=0)).float().to(device)
        sims = (network.module.get_tsm(vis_batch) + 1)/2
        for i in range(2,7):
            fig.add_subplot(2,3,i)
            plt.imshow(sims[i-2], vmin=-1, vmax=2)
            print(np.max(sims[i-2]), np.min(sims[i-2]))
        plt.savefig(f'{VER}.jpg')
        plt.clf()

        print(f"VAL - EPOCH {epoch}")
        network.eval()

        val_dict = {}
        for j, (feature, filenames, durations) in enumerate(tqdm(validation_dataloader)):
            feature = feature.to(device)
            with torch.no_grad():
                tsm = network(feature)
                mean_tsm = torch.mean(tsm, dim=1)
                boundaries = []
                for i, x in enumerate(mean_tsm):
                    boundary = sorted(f(x, 0))
                    boundaries.append(boundary)
            durations = durations.numpy()
            for i, boundary in enumerate(boundaries):
                duration = durations[i]
                filename = filenames[i]
                boundary_list = []
                first = TIME_UNIT/2
                for j in boundary:
                    if first + TIME_UNIT*j < duration:
                        boundary_list.append(first + TIME_UNIT*j)
                val_dict[filename] = boundary_list

        f1, prec, rec = validate(val_dict)
        print(f'epoch: {epoch}')
        print(f'f1: {f1}')
        print(f'precision: {prec}')
        print(f'recall: {rec}')
        if f1 > best_f1:
            print("* best f1 *")
            best_f1 = f1
            prec_for_best_f1 = prec
            rec_for_best_f1 = rec
            epoch_for_best_f1 = epoch
            best_f1_net = copy.deepcopy(network)
            if SAVE_MODEL:
                best_f1_net_state = network.state_dict()
                torch.save(best_f1_net_state, os.path.join(model_path, f'best_f1_{int(best_f1*10000)}_e{epoch}.pt'))
            # for early stop
            improve_flag = True
            no_improvement_duration = 0
            
        print(f'max_f1_value: {best_f1}')

    print()
    print("== VAL BEST F1 RESULT ==")
    print(f"epoch: {epoch_for_best_f1}")
    print(f"f1: {best_f1}")
    print(f"prec: {prec_for_best_f1}")
    print(f"rec: {rec_for_best_f1}")
    print()
    
    # Test
    print(f"TEST for best f1")
    network = copy.deepcopy(best_f1_net)
    network.eval() 

    test_dict = {}
    for j, (feature, filenames, durations) in enumerate(tqdm(test_dataloader)):
        feature = feature.to(device)
        with torch.no_grad():
            tsm = network(feature)
            mean_tsm = torch.mean(tsm, dim=1)
            boundaries = []
            for i, x in enumerate(mean_tsm):
                boundary = sorted(f(x, 0))
                boundaries.append(boundary)
        durations = durations.numpy()
        for i, boundary in enumerate(boundaries):
            duration = durations[i]
            filename = filenames[i]
            boundary_list = []
            first = TIME_UNIT/2
            for j in boundary:
                if first + TIME_UNIT*j < duration:
                    boundary_list.append(first + TIME_UNIT*j)
            test_dict[filename] = boundary_list

    f1, prec, rec = validate(test_dict, mode='test')
    print(f'epoch: {epoch_for_best_f1}')
    print(f'f1: {f1}')
    print(f'precision: {prec}')
    print(f'recall: {rec}')

