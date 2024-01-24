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

import intronets_format
import intronets_seriation
import intronets_hdf
from intronets_format import *
from intronets_seriation import *
from intronets_hdf import *

from concurrent.futures import ProcessPoolExecutor as Pool



def get_matrices(vcf_file, bed_file, chr_nr=None, polymorphisms=128, stepsize=16, fixed_nr=True, remove_samples_wo_introgression=False, return_also_pos = True, return_for_pytorch=True):
    pop_df_ref, pop_df_target, intro_df_ref, intro_df_target, newsamples_ref, newsamples_target, newpos = format_from_vcf_df(vcf_file, bed_file)
    
    pop_df_ref_genotype = pop_df_ref[["genotype"]].to_numpy().squeeze()
    pop_df_target_genotype = pop_df_target[["genotype"]].to_numpy().squeeze()
    

    
    intro_df_ref_genotype = intro_df_ref[["genotype"]].to_numpy().squeeze()
    intro_df_target_genotype = intro_df_target[["genotype"]].to_numpy().squeeze()
    
    ref_samples_haplos = pop_df_ref[["ind", "hap"]].to_numpy().squeeze()
    target_samples_haplos = pop_df_target[["ind", "hap"]].to_numpy().squeeze()
    
    pop_df_ref_genotype = np.stack(pop_df_ref_genotype, axis=0)
    pop_df_target_genotype = np.stack(pop_df_target_genotype, axis=0)
    intro_df_ref_genotype = np.stack(intro_df_ref_genotype, axis=0)
    intro_df_target_genotype = np.stack(intro_df_target_genotype, axis=0)


    full_array = []
    if polymorphisms != None:
        startpos = 0
        while (startpos + polymorphisms) <=  (pop_df_ref_genotype.shape[1]):
       
            endpos = startpos + polymorphisms
            pop_df_ref_genotype_win = pop_df_ref_genotype[:,startpos:endpos]
            pop_df_target_genotype_win = pop_df_target_genotype[:,startpos:endpos]
            intro_df_ref_genotype_win = intro_df_ref_genotype[:,startpos:endpos]
            intro_df_target_genotype_win = intro_df_target_genotype[:,startpos:endpos]
            positions = [startpos, endpos]
            
            if remove_samples_wo_introgression == True:

                if 1 not in np.unique(intro_df_target_genotype_win):
                    #print("a window without introgression")
                    startpos = startpos + stepsize
                    continue
            
            
            if chr_nr == None:
                full_array.append([pop_df_ref_genotype_win, pop_df_target_genotype_win,intro_df_ref_genotype_win, intro_df_target_genotype_win, positions, ref_samples_haplos, target_samples_haplos ])
            else:
                full_array.append([pop_df_ref_genotype_win, pop_df_target_genotype_win,intro_df_ref_genotype_win, intro_df_target_genotype_win, positions, ref_samples_haplos, target_samples_haplos, chr_nr ])

            
            startpos = startpos + stepsize

    return full_array




def get_matrices_multiproc(files, polymorphisms=128, stepsize=16, fixed_nr=True, random_reg=False, random_el=1, remove_samples_wo_introgression=False, only_first=False, return_also_pos = True, return_for_pytorch=True):
    vcf_file = files[0]
    bed_file = files[1]
    chr_nr = files[2]
    
    pop_df_ref, pop_df_target, intro_df_ref, intro_df_target, newsamples_ref, newsamples_target, newpos = format_from_vcf_df(vcf_file, bed_file)
    
    pop_df_ref_genotype = pop_df_ref[["genotype"]].to_numpy().squeeze()
    pop_df_target_genotype = pop_df_target[["genotype"]].to_numpy().squeeze()
    

    
    intro_df_ref_genotype = intro_df_ref[["genotype"]].to_numpy().squeeze()
    intro_df_target_genotype = intro_df_target[["genotype"]].to_numpy().squeeze()
    
    ref_samples_haplos = pop_df_ref[["ind", "hap"]].to_numpy().squeeze()
    target_samples_haplos = pop_df_target[["ind", "hap"]].to_numpy().squeeze()
    

    pop_df_ref_genotype = np.stack(pop_df_ref_genotype, axis=0)
    pop_df_target_genotype = np.stack(pop_df_target_genotype, axis=0)
    intro_df_ref_genotype = np.stack(intro_df_ref_genotype, axis=0)
    intro_df_target_genotype = np.stack(intro_df_target_genotype, axis=0)
    

    newpos = np.stack(newpos, axis=0)
    
    
    full_array = []

    if random_reg == False:
        if polymorphisms != None:
            startpos = 0
            while (startpos + polymorphisms) <=  (pop_df_ref_genotype.shape[1]):
                
                endpos = startpos + polymorphisms
                
                pop_df_ref_genotype_win = pop_df_ref_genotype[:,startpos:endpos]
                pop_df_target_genotype_win = pop_df_target_genotype[:,startpos:endpos]
                intro_df_ref_genotype_win = intro_df_ref_genotype[:,startpos:endpos]
                intro_df_target_genotype_win = intro_df_target_genotype[:,startpos:endpos]
                positions = [startpos, endpos]

                newpos_win = newpos[startpos:endpos]


                
                
                if remove_samples_wo_introgression == True:

                    if 1 not in np.unique(intro_df_target_genotype_win):
                        #print("a window without introgression")
                        startpos = startpos + stepsize

                        if only_first == True:
                            break
                        continue
                

                if return_also_pos == False:
                    if chr_nr == None:
                        full_array.append([pop_df_ref_genotype_win, pop_df_target_genotype_win,intro_df_ref_genotype_win, intro_df_target_genotype_win, positions, ref_samples_haplos, target_samples_haplos ])
                    else:
                        full_array.append([pop_df_ref_genotype_win, pop_df_target_genotype_win,intro_df_ref_genotype_win, intro_df_target_genotype_win, positions, ref_samples_haplos, target_samples_haplos, chr_nr ])
                else:
                    if chr_nr == None:
                        full_array.append([pop_df_ref_genotype_win, pop_df_target_genotype_win,intro_df_ref_genotype_win, intro_df_target_genotype_win, positions, ref_samples_haplos, target_samples_haplos, newpos_win ])
                    else:
                        full_array.append([pop_df_ref_genotype_win, pop_df_target_genotype_win,intro_df_ref_genotype_win, intro_df_target_genotype_win, positions, ref_samples_haplos, target_samples_haplos, chr_nr, newpos_win ])
                
                startpos = startpos + stepsize

                if only_first == True:
                    break
    
    else:
        for ir in range(random_el):
            geno_length = pop_df_ref_genotype.shape[1]


            if polymorphisms > geno_length:
                print("the haplotype/chromosome is too short!")
                continue

            if polymorphisms == geno_length:
                startpos = 0
            else:
                startpos = np.random.randint(0, geno_length - polymorphisms)
            endpos = startpos + polymorphisms


            pop_df_ref_genotype_win = pop_df_ref_genotype[:,startpos:endpos]
            pop_df_target_genotype_win = pop_df_target_genotype[:,startpos:endpos]
            intro_df_ref_genotype_win = intro_df_ref_genotype[:,startpos:endpos]
            intro_df_target_genotype_win = intro_df_target_genotype[:,startpos:endpos]
            positions = [startpos, endpos]

            #newpos window
            newpos_win = newpos[startpos:endpos]
            
            
            if remove_samples_wo_introgression == True:

                if 1 not in np.unique(intro_df_target_genotype_win):
                    #print("a window without introgression")
                    startpos = startpos + stepsize

                    if only_first == True:
                        break
                    continue
            
            

            if return_also_pos == False:
                if chr_nr == None:
                    full_array.append([pop_df_ref_genotype_win, pop_df_target_genotype_win,intro_df_ref_genotype_win, intro_df_target_genotype_win, positions, ref_samples_haplos, target_samples_haplos ])
                else:
                    full_array.append([pop_df_ref_genotype_win, pop_df_target_genotype_win,intro_df_ref_genotype_win, intro_df_target_genotype_win, positions, ref_samples_haplos, target_samples_haplos, chr_nr ])
            else:
                if chr_nr == None:
                    full_array.append([pop_df_ref_genotype_win, pop_df_target_genotype_win,intro_df_ref_genotype_win, intro_df_target_genotype_win, positions, ref_samples_haplos, target_samples_haplos, newpos_win ])
                else:
                    full_array.append([pop_df_ref_genotype_win, pop_df_target_genotype_win,intro_df_ref_genotype_win, intro_df_target_genotype_win, positions, ref_samples_haplos, target_samples_haplos, chr_nr, newpos_win ])


            if only_first == True:
                    break

    return full_array




def process_vcf_df_windowed(folder, polymorphisms=128, stepsize=16, one_target=True):
    vcf_files, bed_files = get_vcf_bed_folder(folder)

    
    all_entries = []

    counter = 0
    for vcf_file, bed_file in zip(vcf_files, bed_files):

        full_array = get_matrices(vcf_file, bed_file, chr_nr=counter, polymorphisms=polymorphisms, stepsize=stepsize)
        counter = counter + 1
        
        if len(full_array) == 0:
            continue
        all_entries.append(full_array)

    
    final_array = []
    for entry in all_entries:
        for subentry in entry:
            pop_target_sorted, pop_ref_sorted, intro_target_sorted, intro_ref_sorted, samples_target_sorted, samples_ref_sorted = apply_lsum_and_seriation_and_sort(subentry[1], subentry[0],subentry[3],ind_array1=subentry[6], ind_array2=subentry[5], target_array2=subentry[2])

            
            final_array.append([pop_target_sorted, pop_ref_sorted, intro_target_sorted, intro_ref_sorted, samples_target_sorted, samples_ref_sorted, subentry[4], subentry[7]])


    return final_array  



def process_vcf_df_windowed_multiproc(folder, polymorphisms=128, stepsize=16, start_rep=0, only_first=False, random_reg=False, random_el=1, ignore_zero_introgression=True, one_target=True):
    vcf_files, bed_files = get_vcf_bed_folder(folder, ignore_zero_introgression = ignore_zero_introgression)
    
    from functools import partial

    
    all_entries = []
    
    pool = Pool()

    #the counters-list indicates the replicate / file
    counters = list(range(start_rep, start_rep + len(vcf_files)))
    
    files = list(zip(vcf_files, bed_files, counters))
    
    #the windows are extracted in parallel
    get_matrices_call=partial(get_matrices_multiproc, polymorphisms=polymorphisms, stepsize=stepsize, remove_samples_wo_introgression=ignore_zero_introgression, random_reg=random_reg, random_el=random_el, only_first=only_first)
    
    
    full_array =  pool.map(get_matrices_call, files) 
    

    all_entries = full_array

    #pool2 = Pool()
    
    flattened_entries = []
    for entry in all_entries:
        for subentry in entry:

            #subentry[-1] should be positions - this information can be harnessed later
            flattened_entries.append([subentry[1], subentry[0],subentry[3],subentry[6],subentry[5],subentry[2], subentry[4], subentry[7], subentry[-1]])


    #seriation is done in parallel
    final_array = zip(* pool.map(apply_lsum_and_seriation_and_sort_multiproc, flattened_entries) )

    final_array = list(zip(* final_array))

    return final_array  





