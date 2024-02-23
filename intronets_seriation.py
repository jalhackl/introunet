
from scipy.spatial.distance import pdist
from seriate import seriate
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, cdist
from seriate import seriate
import numpy as np
import pandas as pd
import allel
import os
from pathlib import Path



def apply_seriation(genotype_array, metric = 'euclidean'):
    """
    Description:
        function which applies seriation on the input array

    Arguments:
        genotype_array np.array: input array for seriation
        metric str: metric to be used (has to be supported by sklearn)
    """

    x = genotype_array
    Dx = pdist(x, metric = metric)
    Dx[np.where(np.isnan(Dx))] = 0.
    ix = seriate(Dx, timeout = 0)

    return x[ix], ix



def apply_lsum_and_seriation(genotype_array1, genotype_array2, metric = 'euclidean'):
    """
    Description:
        function which applies seriation on the first genotype array and subsequently linear sum assignment

    Arguments:
        genotype_array1 np.array: list containing the following arrays in this order: genotype_array1, genotype_array2, target_array1, ind_array1, ind_array2, target_array2 
        genotype_array2 np.array: if true, also the positions (stored at the last index of the input array) are returned
    """
    
    genotype_array1_seriated, genotype_array1_seriated_indices = apply_seriation(genotype_array1)
    
    x1 = genotype_array1_seriated
    x2 = genotype_array2
    
    D = cdist(x1, x2, metric = metric)
    D[np.where(np.isnan(D))] = 0.

    i, j = linear_sum_assignment(D)
    
    x2 = x2[j,:]
    
    genotype_array2_seriated_indices = j
    
    return x1, x2, genotype_array1_seriated_indices, genotype_array2_seriated_indices


def apply_lsum_and_seriation_and_sort(genotype_array1, genotype_array2,target_array1,ind_array1=None, ind_array2=None, target_array2=None):
    
    x1, x2, genotype_array1_seriated_indices, genotype_array2_seriated_indices =apply_lsum_and_seriation(genotype_array1, genotype_array2)
    

    
    target_array1_sorted = target_array1[genotype_array1_seriated_indices]
    if type(target_array2) != type(None):
        target_array2_sorted = target_array2[genotype_array2_seriated_indices]
    
    if type(ind_array1) != type(None) and type(ind_array2) != type(None):
        ind_array1_sorted = ind_array1[genotype_array1_seriated_indices]
        ind_array2_sorted = ind_array2[genotype_array2_seriated_indices]
        
        
    if type(target_array2) != type(None) and type(ind_array1) != type(None) and type(ind_array2) != type(None):
        return x1, x2, target_array1_sorted, target_array2_sorted, ind_array1_sorted, ind_array2_sorted
    
    if type(target_array2) != type(None):
        return x1, x2, target_array1_sorted, target_array2_sorted
    
    return x1, x2, target_array1_sorted




def apply_lsum_and_seriation_and_sort_multiproc(flattened_array, return_last_element=True):
    """
    Description:
        function for parallelized application of seriation
        after processing, the seriated list of arrays is returned

    Arguments:
        flattened_array list: list containing the following arrays in this order: genotype_array1, genotype_array2, target_array1, ind_array1, ind_array2, target_array2 
        return_last_element bool: if true, also the positions (stored at the last index of the input array) are returned
    """
 
    #subentry[1], subentry[0],subentry[3],ind_array1=subentry[6], ind_array2=subentry[5], target_array2=subentry[2]
    
    genotype_array1 = flattened_array[0]
    genotype_array2 = flattened_array[1]
    target_array1 = flattened_array[2]
    ind_array1 = flattened_array[3]
    ind_array2 = flattened_array[4]
    target_array2 = flattened_array[5]

    x1, x2, genotype_array1_seriated_indices, genotype_array2_seriated_indices =apply_lsum_and_seriation(genotype_array1, genotype_array2)
    

    
    target_array1_sorted = target_array1[genotype_array1_seriated_indices]
    if type(target_array2) != type(None):
        target_array2_sorted = target_array2[genotype_array2_seriated_indices]
    
    if type(ind_array1) != type(None) and type(ind_array2) != type(None):
        ind_array1_sorted = ind_array1[genotype_array1_seriated_indices]
        ind_array2_sorted = ind_array2[genotype_array2_seriated_indices]
        
        
    if type(target_array2) != type(None) and type(ind_array1) != type(None) and type(ind_array2) != type(None):
        #return x1, x2, target_array1_sorted, target_array2_sorted, ind_array1_sorted, ind_array2_sorted
        
        if return_last_element == False:
            return x1, x2, target_array1_sorted, target_array2_sorted, ind_array1_sorted, ind_array2_sorted, flattened_array[6], flattened_array[7]
        else:
            return x1, x2, target_array1_sorted, target_array2_sorted, ind_array1_sorted, ind_array2_sorted, flattened_array[6], flattened_array[7], flattened_array[-1]


    if type(target_array2) != type(None):
        return x1, x2, target_array1_sorted, target_array2_sorted
    
    return x1, x2, target_array1_sorted

        
def only_restructure_flattened_array(flattened_array, return_last_element=True):
    genotype_array1 = flattened_array[0]
    genotype_array2 = flattened_array[1]
    target_array1 = flattened_array[2]
    ind_array1 = flattened_array[3]
    ind_array2 = flattened_array[4]
    target_array2 = flattened_array[5]

    if type(target_array2) != type(None) and type(ind_array1) != type(None) and type(ind_array2) != type(None):
    #return x1, x2, target_array1_sorted, target_array2_sorted, ind_array1_sorted, ind_array2_sorted
    
        if return_last_element == False:
            return genotype_array1, genotype_array2, target_array1, target_array2, ind_array1, ind_array2, flattened_array[6], flattened_array[7]
        else:
            return genotype_array1, genotype_array2, target_array1, target_array2, ind_array1, ind_array2, flattened_array[6], flattened_array[7], flattened_array[-1]


    if type(target_array2) != type(None):
        return genotype_array1, genotype_array2, target_array1, target_array2
    
    return genotype_array1, genotype_array2, target_array1



        
def restrict_region(pop_df_ref_genotype, pop_df_target_genotype, intro_df_ref_genotype, intro_df_target_genotype, positions=None, nr_polymorphisms=128, start_poly=0, end_poly=None, random_reg=False):
    if random_reg == False:
        if start_poly != None: #and end_poly == None:
            
            if end_poly == None:
                end_poly = start_poly + nr_polymorphisms
            

            
    
    else:
        geno_length = len(pop_df_ref_genotype[0])
        
        start_poly = np.random.randint(0, geno_length - nr_polymorphisms)
        end_poly = start_poly + nr_polymorphisms
        
        
    pop_df_ref_genotype = pop_df_ref_genotype[:, start_poly:end_poly ]
    pop_df_target_genotype = pop_df_target_genotype[:, start_poly:end_poly ]

    intro_df_ref_genotype = intro_df_ref_genotype[:, start_poly:end_poly ]
    intro_df_target_genotype = intro_df_target_genotype[:, start_poly:end_poly ]
            
    if type(positions) != type(None):
        positions = positions[start_poly:end_poly]
        return pop_df_ref_genotype, pop_df_target_genotype, intro_df_ref_genotype, intro_df_target_genotype, positions
    
    return pop_df_ref_genotype, pop_df_target_genotype, intro_df_ref_genotype, intro_df_target_genotype
