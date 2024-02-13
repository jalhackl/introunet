import os
import numpy as np
import pandas as pd
from intronets_seriation import *
from intronets_format import *

#These functions are only called if no windowing is used

def full_format_seriate_pipeline_df_multiproc(files, fixed_nr=True, remove_samples_wo_introgression=True, nr_polymorphisms=128, random_restrict=True,return_also_pos = True):

        
    vcf_file = files[0]
    bed_file = files[1]
    
    
    pop_df_ref, pop_df_target, intro_df_ref, intro_df_target, newsamples_ref, newsamples_target, newpos = format_from_vcf_df(vcf_file, bed_file)
    
    pop_df_ref_genotype = pop_df_ref[["genotype"]].to_numpy().squeeze()
    pop_df_target_genotype = pop_df_target[["genotype"]].to_numpy().squeeze()
    

    
    intro_df_ref_genotype = intro_df_ref[["genotype"]].to_numpy().squeeze()
    intro_df_target_genotype = intro_df_target[["genotype"]].to_numpy().squeeze()
    

    #stack
    pop_df_ref_genotype = np.stack(pop_df_ref_genotype, axis=0)
    pop_df_target_genotype = np.stack(pop_df_target_genotype, axis=0)
    intro_df_ref_genotype = np.stack(intro_df_ref_genotype, axis=0)
    intro_df_target_genotype = np.stack(intro_df_target_genotype, axis=0)
    
    if fixed_nr == True:
        pop_df_ref_genotype, pop_df_target_genotype, intro_df_ref_genotype, intro_df_target_genotype, newpos = restrict_region(pop_df_ref_genotype, pop_df_target_genotype, intro_df_ref_genotype, intro_df_target_genotype, newpos, nr_polymorphisms=nr_polymorphisms, random_reg=random_restrict)
    
    if remove_samples_wo_introgression == True:

        if 1 not in np.unique(intro_df_target_genotype):
            print("The array does not contain any introgression, an empty is list is returned...")
            return []
            
    ref_samples_haplos = pop_df_ref[["ind", "hap"]].to_numpy().squeeze()
    target_samples_haplos = pop_df_target[["ind", "hap"]].to_numpy().squeeze()
    
    

    pop_target_sorted, pop_ref_sorted, intro_target_sorted, intro_ref_sorted, samples_target_sorted, samples_ref_sorted = apply_lsum_and_seriation_and_sort(pop_df_target_genotype, pop_df_ref_genotype,intro_df_target_genotype,ind_array1=target_samples_haplos, ind_array2=ref_samples_haplos, target_array2=intro_df_ref_genotype)
    
    
    if return_also_pos == True:
        return pop_target_sorted, pop_ref_sorted, intro_target_sorted, intro_ref_sorted, samples_target_sorted, samples_ref_sorted, newpos
    else:
        return pop_target_sorted, pop_ref_sorted, intro_target_sorted, intro_ref_sorted, samples_target_sorted, samples_ref_sorted




def process_vcf_df_multiproc(folder, fixed_nr= True,remove_samples_wo_introgression=True, polymorphisms=128, random_restrict=True):
    vcf_files, bed_files = get_vcf_bed_folder(folder, ignore_zero_introgression=remove_samples_wo_introgression)
    
    
    all_entries = []
    
        
    import multiprocessing
    from functools import partial

    
    all_entries = []
    
    pool = multiprocessing.Pool()
    
    files = list(zip(vcf_files, bed_files))
    
    full_format_seriate_pipeline_df_call=partial(full_format_seriate_pipeline_df_multiproc, fixed_nr=fixed_nr, nr_polymorphisms=polymorphisms, remove_samples_wo_introgression=remove_samples_wo_introgression, random_restrict=random_restrict)

    
    full_array =  pool.map(full_format_seriate_pipeline_df_call, files) 
    
    for entry in full_array:
        if len(entry) > 0:
            all_entries.append(entry)
        else:
            print("empty array, not enough polymorphisms")


    return all_entries


