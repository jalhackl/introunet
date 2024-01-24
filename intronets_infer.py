import pandas as pd
import pickle
import numpy as np
import h5py

import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 18})

import numpy as np
import logging
import argparse
import os

import torch

import h5py


from layers import *
from data_loaders import H5UDataGenerator

from scipy.special import expit
import pickle

import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score
import pandas as pd
import random
from scipy.interpolate import interp1d
from evaluate_unet_windowed_orig import gaussian




def get_chunk(entries, both_y=False):
    indices = np.array( [entry[4] for entry in entries])
    ix = np.array([entry[-1] for entry in entries])
    x = np.array([np.stack(entry[0:2]) for entry in entries])

    if both_y == False:
        y = np.array([entry[2] for entry in entries])
    else:
        y = np.array([np.stack(entry[2:4]) for entry in entries])
    pos = [entry[-2] for entry in entries]
    startpos = [poss[0] for poss in pos]
    endpos = [poss[1] for poss in pos]

    return x,y,indices,ix,startpos,endpos



def predict_model_intronets(weights, ifile,  net="default", n_classes=1, chunk_size=4, smooth=False, filter_multiplier=1, sigma = 30, return_full = False, row_wise_addition=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    #device = torch.device('cpu')


    if net == "default":
        model = NestedUNet(int(n_classes), 2, filter_multiplier = float(filter_multiplier), small = False)
    elif net == "multi":
        model = NestedUNet(int(n_classes), 3, filter_multiplier = float(filter_multiplier), small = False)
    elif net == "multi_fwbw":
        model = NestedUNet(int(n_classes), 4, filter_multiplier = float(filter_multiplier), small = False)
    elif net == "lstm":
        model = NestedUNetLSTM(int(n_classes), 2, filter_multiplier = float(filter_multiplier), small = False)
    elif net == "gru":
        model = NestedUNetLSTM(int(n_classes), 2, filter_multiplier = float(filter_multiplier), create_gru=True, small = False)
    elif net == "lstm_fwbw":
        model = NestedUNetLSTM_fwbw(int(n_classes), 2, filter_multiplier = float(filter_multiplier), small = False)
    elif net == "gru_fwbw":
        model = NestedUNetLSTM_fwbw(int(n_classes), 2, filter_multiplier = float(filter_multiplier), create_gru=True, small = False)
    elif net == "extra":
        model = NestedUNetExtraPos(int(n_classes), 2, filter_multiplier = float(filter_multiplier), small = False)
    else:
        model = NestedUNet(int(n_classes), 2, filter_multiplier = float(filter_multiplier), small = False)


    if weights != "None":
        checkpoint = torch.load(weights, map_location = device)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()


    filename = ifile
    ifile = h5py.File(ifile, 'r')

    keys = list(ifile.keys())

    key = keys[0]

    poly_nr = ifile[key]["y"].shape[3]
    ind_nr = ifile[key]["y"].shape[2]

    G = gaussian(int(poly_nr), int(sigma))

    G = G.view(1, 1, int(poly_nr)).to(device)
    Gn = G.detach().cpu().numpy()

    #main part
    full_ypred = []
    full_ytrue = []


    haplotype_input = True


    #initialize

    try:
        indices = ifile["0"]["indices"][()][0]
    except:
        print("the h5-file has no indices of the individuals... So only simple evaluation (no windowing) is possible")


    #only take the target indices
    indices_start = indices[0]


    idx = np.lexsort((indices_start[:,0], indices_start[:,1]))
    arr_sorted = indices_start[idx]


    arr_sorted = np.array(pd.DataFrame(arr_sorted).drop_duplicates(keep='first'))

    named_arr = list(range(len(arr_sorted)))


    max_replicate = 0
    max_endpos = 0
    try:
        for key in keys:
            for entry in ifile[key]["ix"][()].flatten():

                if entry > max_replicate:
                    max_replicate = entry

            for pos_entry in ifile[key]["pos"][()]:
                pos = pos_entry

                endpos = [x[0][1] for x in pos][0]


                if endpos > max_endpos:
                    max_endpos = endpos    
        replicate_nr = max_replicate 
    except:
        print("the h5-file has no replicate nr of the individuals... So only simple evaluation (no windowing) is reasonable")
        replicate_nr = (len(ifile.keys()) * chunk_size) - 1


    full_arr_pred = np.zeros((replicate_nr+1, len(arr_sorted), max_endpos), dtype = np.float32)
    full_arr_true = np.zeros((replicate_nr+1, len(arr_sorted), max_endpos), dtype = np.float32)
    full_arr_count = np.zeros((replicate_nr+1,len(arr_sorted), max_endpos), dtype = np.float32)

    int_mapping = dict()
    for i, arr_entry in enumerate(arr_sorted):
        int_mapping[named_arr[i]] = tuple(arr_entry)
    haplo_mapping = {v: k for k, v in int_mapping.items()}


    keys = ifile.keys()

    replicate_counter = 0
    startpos1 = 0
    endpos1 = startpos1 + poly_nr


    counter = 0

    for key in keys:

        x = ifile[key]["x_0"]
        y = ifile[key]["y"]

        #print("all the shapes")
        try:
            ix = ifile[key]["ix"][()]

            replicate = [x[0] for x in ix]
            replicate = np.array(replicate)

        except:
            print("no ix-group in h5f-file!")
            ix = list(range(replicate_counter, replicate_counter+chunk_size))
            replicate_counter = replicate_counter + chunk_size
            replicate = np.array(ix)

        
        try:
            pos = ifile[key]["pos"][()]


            startpos = [x[0][0][0] for x in pos]
            endpos = [x[0][0][1] for x in pos]

        except:
            print("problem with position array!")


        try:
            indices = ifile[key]["indices"]

            indices = [x[0,:] for x in indices]
            indices = np.array(indices)

        except:
            print("problem with indices array")

        
        # single channel case - should not happen with new functions
        if len(y.shape) == 3:
            y = np.expand_dims(y, 1)
        
        
        y_pred_ = []

        x_ = torch.FloatTensor(x).to(device)
        
        with torch.no_grad():
            y_ = model(x_)
            
            if smooth == True:
                y_ = y_ * G
            
        y_ = y_.detach().cpu().numpy()

        if x_.shape[0] == 1:
            y_ = np.expand_dims(y_, 0)
    

        full_ypred.append(y_)
        full_ytrue.append(y)

        
        y_pred_.append(y_)

        #simple compare part

        for i, y_chunk in enumerate(y_):   
     
            indiv_inds = []
            if haplotype_input == True:
                for index in indices[i]:

                    new_index = haplo_mapping[tuple(index)]
                    indiv_inds.append(new_index)
            else:
                for index in indices[i][0]:
                    indiv_inds = indices[0][0][0]


            if row_wise_addition == True:
                for ir, row in enumerate(y_chunk):
                    y_row = y[i][0][ir]
                    full_arr_true[replicate[i][0]][indiv_inds[ir],startpos[i]:endpos[i]]  = y_row
                    full_arr_pred[replicate[i][0]][indiv_inds[ir],startpos[i]:endpos[i]] = full_arr_pred[replicate[i][0]][indiv_inds[ir],startpos[i]:endpos[i]] + row


                    if smooth == False:
                        full_arr_count[replicate[i][0]][indiv_inds[ir],startpos[i]:endpos[i]]  = full_arr_count[replicate[i][0]][indiv_inds[ir],startpos[i]:endpos[i]]  + 1

                    else:
                        full_arr_count[replicate[i][0]][indiv_inds[ir],startpos[i]:endpos[i]]  = full_arr_count[replicate[i][0]][indiv_inds[ir],startpos[i]:endpos[i]]  + Gn.flatten()


            else:
            
                full_arr_true[replicate[i][0]][indiv_inds,startpos[i]:endpos[i]]  = y[i]
                full_arr_pred[replicate[i][0]][indiv_inds,startpos[i]:endpos[i]] = full_arr_pred[replicate[i][0]][indiv_inds,startpos[i]:endpos[i]]  + y_chunk


                if smooth == False:
                    full_arr_count[replicate[i][0]][indiv_inds,startpos[i]:endpos[i]]  = full_arr_count[replicate[i][0]][indiv_inds,startpos[i]:endpos[i]]  + 1

                else:
                    full_arr_count[replicate[i][0]][indiv_inds,startpos[i]:endpos[i]]  = full_arr_count[replicate[i][0]][indiv_inds,startpos[i]:endpos[i]]  + Gn.flatten()
            



    full_ypred_final =expit(np.array(full_ypred).flatten())

    from sklearn.metrics import precision_recall_curve
    precision_simple, recall_simple, thresholds_simple = precision_recall_curve(np.array(full_ytrue).flatten(), full_ypred_final, drop_intermediate=True)
    #weight_folder_name = weights.split('/')[0]
    weight_folder_split = weights.split('/')
    if len(weight_folder_split) == 2:
        weight_folder_name = weight_folder_split[0]
    else:
        weight_folder_name = ""

    plt.plot(recall_simple, precision_simple)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend()
    plt.savefig(filename.split('.')[0] + "_simple_eval_" + weight_folder_name + ".png")
    plt.figure()


    full_arr_true_flattened = full_arr_true.flatten()
    full_arr_pred_flattened= full_arr_pred.flatten()
    full_arr_count_flattened = full_arr_count.flatten()
    full_arr_count_flattened_nonzero_indices = full_arr_count_flattened.nonzero()

    full_arr_true_flattened_nonzero = full_arr_true_flattened[full_arr_count_flattened_nonzero_indices]
    full_arr_pred_flattened_nonzero = full_arr_pred_flattened[full_arr_count_flattened_nonzero_indices]
    full_arr_count_flattened_nonzero = full_arr_count_flattened[full_arr_count_flattened_nonzero_indices]

    final_pred = full_arr_pred_flattened_nonzero 

    final_expit = expit(final_pred/full_arr_count_flattened_nonzero)
    final_expit1 = final_expit

    precision, recall, thresholds = precision_recall_curve(full_arr_true_flattened_nonzero, final_expit, drop_intermediate=True)

    plt.plot(recall, precision)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend()
    plt.savefig(filename.split('.')[0] + "_full_eval" + weight_folder_name + ".png")

    prec_recall_file = filename.split('.')[0] + "_full_eval" + weight_folder_name + ".pickle"
    prec_recall_file_simple = filename.split('.')[0] + "_simple_eval" + weight_folder_name + ".pickle"

    with open(prec_recall_file,'wb') as f: pickle.dump([precision, recall, thresholds], f)

    with open(prec_recall_file_simple,'wb') as f: pickle.dump([precision_simple, recall_simple, thresholds_simple], f)


    if return_full == False:

        return precision, recall

    else:

        return precision, recall, precision_simple, recall_simple, full_ypred, full_ytrue, full_arr_true, full_arr_pred,  full_arr_true_flattened, full_arr_pred_flattened, full_arr_count_flattened, full_arr_count_flattened_nonzero_indices, final_expit, final_expit1







def predict_model_intronets_simple(weights, ifile,  net="default", chunk_size=4, n_classes=1, smooth=False, filter_multiplier=1, sigma = 30):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    #device = torch.device('cpu')

    if net == "default":
        model = NestedUNet(int(n_classes), 2, filter_multiplier = float(filter_multiplier), small = False)
    elif net == "multi":
        model = NestedUNet(int(n_classes), 3, filter_multiplier = float(filter_multiplier), small = False)
    elif net == "multi_fwbw":
        model = NestedUNet(int(n_classes), 4, filter_multiplier = float(filter_multiplier), small = False)
    elif net == "lstm":
        model = NestedUNetLSTM(int(n_classes), 2, filter_multiplier = float(filter_multiplier), small = False)
    elif net == "gru":
        model = NestedUNetLSTM(int(n_classes), 2, filter_multiplier = float(filter_multiplier), create_gru=True, small = False)
    elif net == "lstm_fwbw":
        model = NestedUNetLSTM_fwbw(int(n_classes), 2, filter_multiplier = float(filter_multiplier), small = False)
    elif net == "gru_fwbw":
        model = NestedUNetLSTM_fwbw(int(n_classes), 2, filter_multiplier = float(filter_multiplier), create_gru=True, small = False)
    elif net == "extra":
        model = NestedUNetExtraPos(int(n_classes), 2, filter_multiplier = float(filter_multiplier), small = False)
    else:
        model = NestedUNet(int(n_classes), 2, filter_multiplier = float(filter_multiplier), small = False)

    if weights != "None":
        checkpoint = torch.load(weights, map_location = device)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    Y = []
    Y_pred = []
    indices_ = []

    counter = 0

    filename = ifile
    ifile = h5py.File(ifile, 'r')

    keys = list(ifile.keys())

    key = keys[0]

    poly_nr = ifile[key]["y"].shape[3]
    ind_nr = ifile[key]["y"].shape[2]

    G = gaussian(int(poly_nr), int(sigma))
    G = G.view(1, 1, int(poly_nr)).to(device)
    Gn = G.detach().cpu().numpy()


    #main part
    full_ypred = []
    full_ytrue = []


    try:
        indices = ifile["0"]["indices"][()][0]
    except:
        print("the h5-file has no indices of the individuals... So only simple evaluation (no windowing) is possible")



    max_replicate = 0
    try:
        for key in keys:
            for entry in ifile[key]["ix"][()].flatten():

                if entry > max_replicate:
                    max_replicate = entry
        replicate_nr = max_replicate - 1
    except:
        print("no ix entry")
        replicate_nr = (len(ifile.keys()) * chunk_size) - 1


    keys = ifile.keys()

    replicate_counter = 0
    startpos1 = 0
    endpos1 = startpos1 + poly_nr


    counter = 0
    for key in keys:
        
        if counter > 100:
            break
        counter = counter + 1



        x = ifile[key]["x_0"]
        y = ifile[key]["y"]
        try:
            ix = ifile[key]["ix"]
            replicate = [x[0] for x in ix]
            replicate = [x for xs in replicate for x in xs]


        except:
            ix = list(range(replicate_counter, replicate_counter+chunk_size))
            replicate_counter = replicate_counter + chunk_size
            replicate = ix
        
        try:
            pos = ifile[key]["pos"]
            startpos = [x[0][0][0] for x in pos]
            endpos = [x[0][0][1] for x in pos]
        except:
            startpos = chunk_size * [startpos1]
            endpos = chunk_size * [endpos1]

        try:
            indices = ifile[key]["indices"]

            indices = [x[0,:] for x in indices]
            indices = np.array(indices)
        except:
            indices = list(range(ind_nr))
            indices2 = np.array(list(range(np.max(indices)+1, np.max(indices)+len(indices)+1)))
            indicesfull = np.vstack([indices, indices2])
            indt=np.tile(indicesfull, (chunk_size,1,1))
            indices = np.expand_dims(indt, 1)

        
        # single channel case
        if len(y.shape) == 3:
            y = np.expand_dims(y, 1)

        
        y_pred_ = []

        x = np.array(x)
        x_ = torch.FloatTensor(x).to(device)
        
        with torch.no_grad():
            y_ = model(x_)
            
            if smooth == True:
                y_ = y_ * G
            
        y_ = y_.detach().cpu().numpy()
        if x_.shape[0] == 1:
            y_ = np.expand_dims(y_, 0)
        
        
        y_pred_.append(y_)


        #simple compare part

        full_ypred.append(y_)
        full_ytrue.append(y)


    #simple compare evaluation
    full_ypred_final =expit(np.array(full_ypred).flatten())

    from sklearn.metrics import precision_recall_curve
    precision_simple, recall_simple, thresholds_simple = precision_recall_curve(np.array(full_ytrue).flatten(), full_ypred_final)

    plt.plot(recall_simple, precision_simple)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend()
    plt.savefig(filename.split('.')[0] + "_simple_eval" + ".png")

    return precision_simple, recall_simple








            
