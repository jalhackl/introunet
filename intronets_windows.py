from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, cdist
from seriate import seriate
import numpy as np
import pandas as pd
import allel
from pathlib import Path

from intronets_format import *
from intronets_seriation import *
from intronets_hdf import *

from concurrent.futures import ProcessPoolExecutor as Pool



def get_matrices(vcf_file, bed_file, chr_nr=None, polymorphisms=128, stepsize=16, remove_samples_wo_introgression=False):
    """
    Description:
        returns an array containing - NOT used in the usual workflow, because parallelized version get_matrices_multiproc is used: 
        pop_df_ref_genotype_win: windows of reference haplotype
        pop_df_target_genotype_win: windows of target haplotype
        intro_df_ref_genotype_win: windows of reference introgression (0: no / 1: true); for unidirectional introgression, it only contains 0s
        intro_df_target_genotype_win: windows of target introgression (0: no / 1: true)
        positions: SNP position
        ref_samples_haplos: name of the individuals in the reference population
        target_samples_haplos: name of the individuals in the target population
        chr_nr: if chr_nr!=None, also the replicate number is returned


    Arguments:
        vcf_file str: folder containing files
        bed_file str: folder containing files
    """
        
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




def get_matrices_multiproc(files, polymorphisms=128, stepsize=16, random_reg=False, random_el=1, remove_samples_wo_introgression=False, only_first=False, return_also_pos = True):
    """
    Description:
        returns an array containing:
        pop_df_ref_genotype_win: windows of reference haplotype
        pop_df_target_genotype_win: windows of target haplotype
        intro_df_ref_genotype_win: windows of reference introgression (0: no / 1: true); for unidirectional introgression, it only contains 0s
        intro_df_target_genotype_win: windows of target introgression (0: no / 1: true)
        positions: SNP position
        ref_samples_haplos: name of the individuals in the reference population
        target_samples_haplos: name of the individuals in the target population
        chr_nr: if chr_nr!=None, also the replicate number is returned


    Arguments:
        files list: list containing three arrays: vcf-file information, bed-file information, and nr of the current replicate
        bed_file str: folder containing files
        polymorphisms int: number of polymorphisms to be used for one window
        stepsize int: stepsize of the windowing process
        random_reg bool: if True, one or more (indicated by random_el) windows are randomly cut out
        random_el int: number of randomly chosen windows (if random_reg == True)
        remove_samples_wo_introgression bool: if True, windows without introgression are discarded
        only_first bool: if True, only the first window (i.e. the first SNPs indicated by the polymorphisms argument) is processed
        return_also_pos bool: if True, also the position of the window within the chromsome is returned

    """
        
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




def process_vcf_df_windowed(folder, polymorphisms=128, stepsize=16):
    """
    Description:
        NOT used in the usual workflow, because parallelized version get_matrices_multiproc is used: 
        returns an array containing the following subarrays: 
        pop_df_ref_genotype_win: windows of reference haplotype
        pop_df_target_genotype_win: windows of target haplotype
        intro_df_ref_genotype_win: windows of reference introgression (0: no / 1: true); for unidirectional introgression, it only contains 0s
        intro_df_target_genotype_win: windows of target introgression (0: no / 1: true)
        positions: SNP position
        ref_samples_haplos: name of the individuals in the reference population
        target_samples_haplos: name of the individuals in the target population
        chr_nr: if chr_nr!=None, also the replicate number is returned


    Arguments:
        folder folder: folder containing files
        polymorphisms int: folder containing files
        stepsize int: 
    """
        
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



def process_vcf_df_windowed_multiproc(folder, polymorphisms=128, stepsize=16, start_rep=0, only_first=False, random_reg=False, random_el=1, ignore_zero_introgression=True):
    """
    Description:
        processes all pairs of vcf- and bed- files in all subdirectories of the input folder
        seriation is applied and an array ready to be stored as hdf is returned


    Arguments:
        folder folder: folder containing files (vcf and bed with truth tracts)
        polymorphisms int: number of polymorphisms used for one window
        stepsize int: stepsize for the windowing process
        start_rep int: the replicate number which is added to the first replicate of the input data, i.e. for start_rep=0, the first replicate is named 0, for start_rep=500, 500...
        only_first bool: if True, only one window is cut out of the chromosome
        random_reg bool: if True, the windowing process does not start at base pair 0, but windows are randomly chosen from the chromosome
        random_el int: the number of randomly chosen windows from the chromsome (if random_reg is set to True), i.e. if random_el=1, only one window is cut out
        ignore_zero_introgression bool: if True, only windows with introgression content are considered
    """
        
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
    
    #for further processing, the nested entries (one index per vcf-file) are flattened
    flattened_entries = []
    for entry in all_entries:
        for subentry in entry:

            #subentry[-1] should be positions - this information can be harnessed later
            flattened_entries.append([subentry[1], subentry[0],subentry[3],subentry[6],subentry[5],subentry[2], subentry[4], subentry[7], subentry[-1]])


    #seriation is done in parallel
    final_array = zip(* pool.map(apply_lsum_and_seriation_and_sort_multiproc, flattened_entries) )

    final_array = list(zip(* final_array))

    return final_array  





