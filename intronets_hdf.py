from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, cdist
from seriate import seriate
import numpy as np
import pandas as pd
import allel
#import utils
import os
from pathlib import Path


#import intronets_seriation
#import intronets_format
#import intronets_windows
#from intronets_format import *
#from intronets_seriation import *
#from intronets_windows import *


def final_format(entries, only_target_intro = True, change_ref_target=False, fullpos=True):
    final_entries = []
    for entry in entries:
        if change_ref_target == False:
            x = np.stack([entry[0], entry[1]], axis = 0)
        else:
            x = np.stack([entry[1], entry[0]], axis = 0)
        
        if only_target_intro == True:
            #2nd entry is target
            intro = np.stack(entry[2], axis = 0)
        else:
            intro = np.stack([entry[2], entry[3]], axis = 0)
            
        samples = np.stack([entry[4], entry[5]], axis = 0)
        
        position = entry[6][0]
        endposition = entry[6][1]
        ix = entry[7]

        #positions of polymorphisms
        positions = entry[-1]
        
        if fullpos == True:
            final_entries.append([x, intro, samples, np.array([entry[6]]), endposition, ix, positions])
        else:
            final_entries.append([x, intro, samples, position, endposition, ix, positions])
        
    return final_entries
            

def create_hdf_table_extrakey_chunk3(hdf_file, input_entries, x_name = "x_0", y_name="y", chunk_size=4):
    import h5py
    
    act_shape0 = input_entries[0][0].shape
    act_shape1 = input_entries[0][1].shape
    num_lines = len(input_entries)
    
    with h5py.File(hdf_file, 'w') as h5f:
    
        for i in range(0, len(input_entries)-chunk_size, chunk_size):
            
            dset1 = h5f.create_dataset(str(i) + "/" + x_name,
                                   shape=(chunk_size, act_shape0[0], act_shape0[1], act_shape0[2]),
                                   compression='lzf',
                                   dtype=np.uint8)
            dset2 = h5f.create_dataset(str(i) + "/" + y_name,
                                       shape=(chunk_size, 1, act_shape1[0], act_shape1[1]),
                                       compression='lzf',
                                       dtype=np.uint8)
            
            for k in range(chunk_size):
                entry = input_entries[i+k]


                features = entry[0]
                labels = entry[1]


                dset1[k] = features
                dset2[k] = [labels]



def create_hdf_table_extrakey_chunk3_groups(hdf_file, input_entries, start_nr=0, x_name = "x_0", y_name="y", chunk_size=4):
    #this function takes a start group number as input and returns the final group number (so that the next iteration can start with this number and no groups are overwritten)
    import h5py
    
    act_shape0 = input_entries[0][0].shape
    act_shape1 = input_entries[0][1].shape
    num_lines = len(input_entries)
    
    with h5py.File(hdf_file, 'w') as h5f:
    
        for i in range(0, len(input_entries)-chunk_size+1, chunk_size):
            
            dset1 = h5f.create_dataset(str(i+start_nr) + "/" + x_name,
                                   shape=(chunk_size, act_shape0[0], act_shape0[1], act_shape0[2]),
                                   compression='lzf',
                                   dtype=np.uint8)
            dset2 = h5f.create_dataset(str(i+start_nr) + "/" + y_name,
                                       shape=(chunk_size, 1, act_shape1[0], act_shape1[1]),
                                       compression='lzf',
                                       dtype=np.uint8)
            
            for k in range(chunk_size):
                entry = input_entries[i+k]

                features = entry[0]
                labels = entry[1]

                dset1[k] = features
                dset2[k] = [labels]

     #return the current group number
    return i+start_nr+chunk_size


def create_hdf_table_extrakey_chunk3_windowed(hdf_file, input_entries, start_nr=0, x_name = "x_0", y_name="y", ind_name="indices", pos_name="pos", ix_name="ix", chunk_size=4):
    import h5py
    
    input_entries = final_format(input_entries)
    
    act_shape0 = input_entries[0][0].shape
    act_shape1 = input_entries[0][1].shape
    
    act_shape2 = input_entries[0][2].shape
    act_shape3 = input_entries[0][3].shape
    
    #with h5py.File(hdf_file, 'w') as h5f:
    #create if not existent, otherwise add entries
    with h5py.File(hdf_file, 'a') as h5f:

    # use num_features-1 if the csv file has a column header
        for i in range(0, len(input_entries)-chunk_size+1, chunk_size):
            
            dset1 = h5f.create_dataset(str(i+start_nr) + "/" + x_name,
                                   shape=(chunk_size, act_shape0[0], act_shape0[1], act_shape0[2]),
                                   compression='lzf',
                                   dtype=np.uint8)
            dset2 = h5f.create_dataset(str(i+start_nr) + "/" + y_name,
                                       shape=(chunk_size, 1, act_shape1[0], act_shape1[1]),
                                       compression='lzf',
                                       dtype=np.uint8)
            
            dset3 = h5f.create_dataset(str(i+start_nr) + "/" + ind_name,
                                       shape=(chunk_size, act_shape2[0], act_shape2[1], act_shape2[2]),
                                       compression='lzf',
                                       dtype=np.uint32)
            
            dset4 = h5f.create_dataset(str(i+start_nr) + "/" + pos_name,
                                       shape=(chunk_size, 1, act_shape3[0], act_shape3[1]),
                                       compression='lzf',
                                       dtype=np.uint32)

            dset5 = h5f.create_dataset(str(i+start_nr) + "/" + ix_name,
                            shape=(chunk_size, 1, 1),
                            compression='lzf',
                            dtype=np.uint32)
            
            
            for k in range(chunk_size):
                entry = input_entries[i+k]

                features = entry[0]
                labels = entry[1]
                

                dset1[k] = features
                dset2[k] = [labels]
                
                dset3[k] = entry[2]
                dset4[k] = entry[3]

                dset5[k] = entry[5]
    
    #return the current group number
    return i+start_nr+chunk_size


def create_hdf_table_extrakey_chunk3_poschannel(hdf_file, input_entries, x_name = "x_0", y_name="y", chunk_size=4):
    import h5py
    
    act_shape0 = input_entries[0][0].shape
    act_shape1 = input_entries[0][1].shape
    num_lines = len(input_entries)
    
    with h5py.File(hdf_file, 'w') as h5f:
    
        for i in range(0, len(input_entries)-chunk_size, chunk_size):
            
            dset1 = h5f.create_dataset(str(i) + "/" + x_name,
                                   shape=(chunk_size, act_shape0[0]+1, act_shape0[1], act_shape0[2]),
                                   compression='lzf',
                                   dtype=np.uint32)
            dset2 = h5f.create_dataset(str(i) + "/" + y_name,
                                       shape=(chunk_size, 1, act_shape1[0], act_shape1[1]),
                                       compression='lzf',
                                       dtype=np.uint8)
            
            for k in range(chunk_size):
                entry = input_entries[i+k]

                features = entry[0]
                labels = entry[1]
                
                onepos = entry[-1]

                oneposbroad = np.broadcast_to(onepos, (1, features.shape[1], features.shape[2]))
                features = np.concatenate([features, oneposbroad])


                dset1[k] = features         
                dset2[k] = [labels]




def create_hdf_table_extrakey_chunk3_poschannel_scaled(hdf_file, input_entries, x_name = "x_0", y_name="y", chunk_size=4):
    import h5py
    
    act_shape0 = input_entries[0][0].shape
    act_shape1 = input_entries[0][1].shape
    num_lines = len(input_entries)
    
    with h5py.File(hdf_file, 'w') as h5f:
    
        for i in range(0, len(input_entries)-chunk_size, chunk_size):
            
            dset1 = h5f.create_dataset(str(i) + "/" + x_name,
                                   shape=(chunk_size, act_shape0[0]+1, act_shape0[1], act_shape0[2]),
                                   compression='lzf',
                                   dtype=np.uint32)
            dset2 = h5f.create_dataset(str(i) + "/" + y_name,
                                       shape=(chunk_size, 1, act_shape1[0], act_shape1[1]),
                                       compression='lzf',
                                       dtype=np.uint8)
            
            for k in range(chunk_size):
                entry = input_entries[i+k]

                features = entry[0]
                labels = entry[1]
                
                onepos = entry[2]/50000
                oneposbroad = np.broadcast_to(onepos, (1, features.shape[1], features.shape[2]))
                features = np.concatenate([features, oneposbroad])

                dset1[k] = features
                dset2[k] = [labels]



def create_hdf_table_extrakey_chunk3_poschannel_gradient(hdf_file, input_entries, x_name = "x_0", y_name="y", chunk_size=4):
    import h5py
    
    act_shape0 = input_entries[0][0].shape
    act_shape1 = input_entries[0][1].shape
    num_lines = len(input_entries)
    
    with h5py.File(hdf_file, 'w') as h5f:
    
        for i in range(0, len(input_entries)-chunk_size, chunk_size):
            
            dset1 = h5f.create_dataset(str(i) + "/" + x_name,
                                   shape=(chunk_size, act_shape0[0]+1, act_shape0[1], act_shape0[2]),
                                   compression='lzf',
                                   dtype=float)
            dset2 = h5f.create_dataset(str(i) + "/" + y_name,
                                       shape=(chunk_size, 1, act_shape1[0], act_shape1[1]),
                                       compression='lzf',
                                       dtype=np.uint8)
            
            for k in range(chunk_size):
                entry = input_entries[i+k]

                features = entry[0]
                labels = entry[1]
                
                onepos = entry[2]
                onepos = np.gradient(onepos)
                oneposbroad = np.broadcast_to(onepos, (1, features.shape[1], features.shape[2]))
                features = np.concatenate([features, oneposbroad])


                dset1[k] = features
                dset2[k] = [labels]



def create_hdf_table_extrakey_chunk3_poschannels_forward_backward(hdf_file, input_entries, x_name = "x_0", y_name="y", chunk_size=4, polymorphisms=128):
    import h5py
    
    act_shape0 = input_entries[0][0].shape
    act_shape1 = input_entries[0][1].shape
    num_lines = len(input_entries)
    
    with h5py.File(hdf_file, 'w') as h5f:
    
        for i in range(0, len(input_entries)-chunk_size, chunk_size):
            
            dset1 = h5f.create_dataset(str(i) + "/" + x_name,
                                   shape=(chunk_size, act_shape0[0]+2, act_shape0[1], act_shape0[2]),
                                   compression='lzf',
                                   dtype=np.uint32)
            dset2 = h5f.create_dataset(str(i) + "/" + y_name,
                                       shape=(chunk_size, 1, act_shape1[0], act_shape1[1]),
                                       compression='lzf',
                                       dtype=np.uint8)
            
            for k in range(chunk_size):
                entry = input_entries[i+k]

                features = entry[0]
                labels = entry[1]
                
                onepos = entry[2]
                onepos = np.gradient(onepos)
                oneposbroad = np.broadcast_to(onepos, (1, features.shape[1], features.shape[2]))
                features = np.concatenate([features, oneposbroad])


                dset1[k] = features
                dset2[k] = [labels]


