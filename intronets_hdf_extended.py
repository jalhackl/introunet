from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, cdist
from seriate import seriate
import numpy as np
import pandas as pd
from pathlib import Path
from intronets_hdf import *

#This file contains 'extend' h5-creation functions (i.e. also storing the distances between SNPs)

def create_hdf_table_extrakey_chunk3_windowed_poschannel(hdf_file, input_entries, start_nr=0, translate_start_to_zero=True, divide_by_seq_length = False, x_name = "x_0", y_name="y", ind_name="indices", pos_name="pos", ix_name="ix", chunk_size=4, seq_length=50000):
    import h5py
    
    input_entries = final_format(input_entries)
    
    act_shape0 = input_entries[0][0].shape
    act_shape1 = input_entries[0][1].shape
    
    act_shape2 = input_entries[0][2].shape
    act_shape3 = input_entries[0][3].shape
    

    #with h5py.File(hdf_file, 'w') as h5f:
    #create if not existent, otherwise add entries
    with h5py.File(hdf_file, 'a') as h5f:

        for i in range(0, len(input_entries)-chunk_size+1, chunk_size):
            
            dset1 = h5f.create_dataset(str(i+start_nr) + "/" + x_name,
                                   shape=(chunk_size, act_shape0[0]+1, act_shape0[1], act_shape0[2]),
                                   compression='lzf',
                                   dtype=np.uint32)
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

                startposition = entry[3]
                endposition = entry[4]

                #positions should be -1 now
                onepos = entry[-1]


                if translate_start_to_zero == True:
                    onepos = onepos - onepos[0]

                if divide_by_seq_length == True:
                    onepos = onepos / seq_length

                oneposbroad = np.broadcast_to(onepos, (1, features.shape[1], features.shape[2]))
                features = np.concatenate([features, oneposbroad])

                dset1[k] = features
                
                dset2[k] = [labels]
                
                dset3[k] = entry[2]
                dset4[k] = entry[3]

                dset5[k] = entry[5]
    
    #return the current group number
    return i+start_nr+chunk_size





def create_hdf_table_extrakey_chunk3_windowed_gradient(hdf_file, input_entries, start_nr=0, x_name = "x_0", y_name="y", ind_name="indices", pos_name="pos", ix_name="ix", chunk_size=4):
    import h5py
    
    input_entries = final_format(input_entries)
    
    act_shape0 = input_entries[0][0].shape
    act_shape1 = input_entries[0][1].shape
    
    act_shape2 = input_entries[0][2].shape
    act_shape3 = input_entries[0][3].shape

    #create if not existent, otherwise add entries
    with h5py.File(hdf_file, 'a') as h5f:

        for i in range(0, len(input_entries)-chunk_size+1, chunk_size):
            
            dset1 = h5f.create_dataset(str(i+start_nr) + "/" + x_name,
                                   shape=(chunk_size, act_shape0[0]+1, act_shape0[1], act_shape0[2]),
                                   compression='lzf',
                                   dtype=float)
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
                

                #positions should be -1 now
                onepos = entry[-1]
                onepos = np.gradient(onepos)
                oneposbroad = np.broadcast_to(onepos, (1, features.shape[1], features.shape[2]))
                features = np.concatenate([features, oneposbroad])

                dset1[k] = features
                dset2[k] = [labels]
                
                dset3[k] = entry[2]
                dset4[k] = entry[3]

                dset5[k] = entry[5]
    
    #return the current group number
    return i+start_nr+chunk_size





def create_hdf_table_extrakey_chunk3_windowed_forward_backward(hdf_file, input_entries, start_nr=0, x_name = "x_0", y_name="y", ind_name="indices", pos_name="pos", ix_name="ix", chunk_size=4, polymorphisms=128):
    import h5py
    
    input_entries = final_format(input_entries)
    
    act_shape0 = input_entries[0][0].shape
    act_shape1 = input_entries[0][1].shape
    
    act_shape2 = input_entries[0][2].shape
    act_shape3 = input_entries[0][3].shape

    
    #with h5py.File(hdf_file, 'w') as h5f:
    #create if not existent, otherwise add entries
    with h5py.File(hdf_file, 'a') as h5f:

        for i in range(0, len(input_entries)-chunk_size+1, chunk_size):
            
            dset1 = h5f.create_dataset(str(i+start_nr) + "/" + x_name,
                                   shape=(chunk_size, act_shape0[0]+2, act_shape0[1], act_shape0[2]),
                                   compression='lzf',
                                   dtype=np.uint32)
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
                
                #positions should be -1 now
                onepos = entry[-1]

                startposition = entry[3]
                endposition = entry[4]

                diff1 = np.diff(onepos, prepend=[onepos[0]])
                diff2 = np.diff(onepos, append=[onepos[-1]])
                
                oneposbroad1 = np.broadcast_to(diff1, (1, features.shape[1], features.shape[2]))
                oneposbroad2 = np.broadcast_to(diff2, (1, features.shape[1], features.shape[2]))

                
                features = np.concatenate([features, oneposbroad1, oneposbroad2])

                dset1[k] = features
                
                dset2[k] = [labels]
                
                dset3[k] = entry[2]
                dset4[k] = entry[3]

                dset5[k] = entry[5]
    
    #return the current group number
    return i+start_nr+chunk_size



