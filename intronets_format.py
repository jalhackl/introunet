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

#import intronets_windows
#import intronets_seriation
#import intronets_hdf
#from intronets_seriation import *
#from intronets_windows import *
#from intronets_hdf import *




def get_vcf_bed_files(folder, ignore_zero_introgression = True):
    """
    Description:
        this function returns the vcf and bed files of the input folder

    Arguments:
        folder str: folder containing files
        ignore_zero_introgression bool: if True, replicates without introgression are discarded
    """
    
    vcf_files = []
    bed_files = []

    all_files = os.listdir(folder)
        
    for file in all_files:
        if file.endswith(".vcf"):
            stem_file = Path(file).stem

            bed_file = stem_file + ".truth.tracts.bed"
            
            if ignore_zero_introgression == True:
                if os.stat(os.path.join(folder,  bed_file)).st_size == 0:
                    continue

            vcf_files.append(os.path.join(folder,  file))
            bed_files.append(os.path.join(folder,  bed_file))
                

    return vcf_files, bed_files


def get_vcf_bed_folder(folder, ignore_zero_introgression = True):
    """
    Description:
        this function returns the vcf and bed files of all subdirectories of the input folder

    Arguments:
        folder str: folder containing files
        ignore_zero_introgression bool: if True, replicates without introgression are discarded
    """
    
    dir_list = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder,d))]  
    
    vcf_files = []
    bed_files = []
    
    for dirr in dir_list:
        all_files = os.listdir(os.path.join(folder,dirr))
        
        for file in all_files:
            if file.endswith(".vcf"):
                stem_file = Path(file).stem
                
                bed_file = stem_file + ".truth.tracts.bed"
                
                if ignore_zero_introgression == True:
                    if os.stat(os.path.join(folder, dirr, bed_file)).st_size == 0:
                        #print("size zero ignored")
                        continue
                
                vcf_files.append(os.path.join(folder, dirr, file))
                bed_files.append(os.path.join(folder, dirr, bed_file))
                

    return vcf_files, bed_files




def format_from_vcf_df(vcf_file, bed_file, return_all=True, upsample=112/2, uniform_upsampling=True, ploidy=2, ref_size=50, target_size=50, select_populations=True, merge_haplotypes=False, compute_scaled_positions=False, fragment_length=50000, add_same_number=False):
    """
    Description:
        loading formatting of the input vcf and bed file 
        lists containing the haplotype arrays, the introgression arrays and the positions are returned
        if return_all == True (default), the following dataframes are returned:
        pop_df_ref: reference haplotypes
        pop_df_target: target haplotypes
        intro_df_ref: contains information if reference haplotype is introgressed (0: no / 1: true); for unidirectional introgression, it only contains 0s
        intro_df_target: contains information if target haplotype is introgressed (0: no / 1: true)
        newsamples_ref: name of the individuals in the reference population
        newsamples_target: name of the individuals in the target population
        newpos: position of the haplotype


    Arguments:
        vcf_file str: folder containing files
        bed_file str: folder containing files
    """
    
    
    #'upsample' indicates the total number of individuals per group (reference / target)
    if upsample != None:
        upsample = upsample * ploidy
        upsample_per_group = upsample
        upsample = upsample * 2
    
    allel_vcf = allel.read_vcf(vcf_file)
    
    newar = allel_vcf["calldata/GT"]
    newpos = allel_vcf["variants/POS"]
    newsamples = allel_vcf["samples"]
    
    
    newar_alt = np.swapaxes(newar.T, 0, 1)

    
    newsamples_int = [[int(s) for s in newsample.split("_") if s.isdigit()] for newsample in newsamples]
    newsamples_int = [item for sublist in newsamples_int for item in sublist]
    
    newsamples_ref = newsamples_int[0:ref_size]
    newsamples_target = newsamples_int[ref_size:]
    
    
    true_tract_data = pd.read_csv(bed_file, sep="\t", header=None, names=['chr', 'start', 'end', 'sample'])


    true_tract_data["hap"] = true_tract_data["sample"].str.split("_").str[-1].astype(int)
    true_tract_data["hap"] = true_tract_data["hap"] - 1

    true_tract_data["ind"] = true_tract_data["sample"].str.split("_").str[-2]

    true_tract_data.drop(columns=['sample'], inplace=True)

    true_tract_data_arr = true_tract_data.to_numpy(dtype=int)

    
    intro_array2 = np.zeros_like(newar_alt)
    
    pop_list = []
    target_list = []

    
    for ie, entry in enumerate(newar_alt):
        curr_sample = newsamples_int[ie]
        
        for i in range(ploidy):
            

            pop_list.append([curr_sample, i, entry[i]])
    
    for entry in true_tract_data_arr:

        curr_ind = entry[-1]
        curr_haplo = entry[-2]

        curr_start = entry[1]
        curr_end = entry[2]
        
        

        for ipos, pos in enumerate(newpos):
            if pos >= curr_start and pos < curr_end:
                
                intro_array2[curr_ind][curr_haplo][ipos] = 1
                

    for ie, entry in enumerate(intro_array2):
        curr_sample = newsamples_int[ie]
        for i in range(ploidy):
            target_list.append([curr_sample, i, entry[i]])
                
    
    pop_df = pd.DataFrame(pop_list, columns=["ind", "hap", "genotype"])
    intro_df = pd.DataFrame(target_list, columns=["ind", "hap", "genotype"])
    
    if select_populations == True:
        pop_df_ref = pop_df[pop_df['ind'].isin(newsamples_ref)]
        pop_df_target = pop_df[pop_df['ind'].isin(newsamples_target)]
        
        intro_df_ref = intro_df[intro_df['ind'].isin(newsamples_ref)]
        intro_df_target = intro_df[intro_df['ind'].isin(newsamples_target)]

        
        if upsample != None:
            total_length = len(pop_df_ref) + len(pop_df_target)
            
            additional_inds = upsample - total_length
            

            
            if additional_inds > 0:

                
                if uniform_upsampling == True:

                    

                    if add_same_number == True:
                        additional_inds_half = additional_inds / 2

                        if (additional_inds % 2) == 1:
                            print("The number of upsample individuals is odd - uniform upsampling and adding the same number to both groups is not possible")
                            return

                        pop1_random = np.random.randint(len(pop_df_ref), size=int(additional_inds_half))
                        pop2_random = np.random.randint(len(pop_df_target), size=int(additional_inds_half))

                    else:
                        reference_length = len(pop_df_ref)
                        target_length = len(pop_df_target)

                        reference_add = np.maximum(upsample_per_group - reference_length, 0)
                        target_add = np.maximum(upsample_per_group - target_length, 0)


                        pop1_random = np.random.randint(reference_length, size=int(reference_add))
                        pop2_random = np.random.randint(target_length, size=int(target_add))

                else:
                    pop_random = np.random.randint(2, size=int(additional_inds))
                    counts_zero_one = np.unique(pop_random, return_counts=True)

                                             
                    additional_inds_ref = counts_zero_one[0]
                    additional_inds_target = counts_zero_one[1]                             
                    
                    pop1_random = np.random.randint(len(pop_df_ref), size=int(additional_inds_ref))
                    pop2_random = np.random.randint(len(pop_df_target), size=int(additional_inds_target))
                    
                                                    
                for nr in pop1_random:
                    new_row = pd.DataFrame(pop_df_ref.iloc[[nr]])
                    pop_df_ref = pop_df_ref._append(new_row, ignore_index = True)

                    new_row = pd.DataFrame(intro_df_ref.iloc[[nr]])
                    intro_df_ref = intro_df_ref._append(new_row, ignore_index = True)


                for nr in pop2_random:
                    new_row = pd.DataFrame(pop_df_target.iloc[[nr]])
                    pop_df_target = pop_df_target._append(new_row, ignore_index = True)


                    new_row = pd.DataFrame(intro_df_target.iloc[[nr]])
                    intro_df_target = intro_df_target._append(new_row, ignore_index = True)
                    

        
        
        if return_all == True:
            return pop_df_ref, pop_df_target, intro_df_ref, intro_df_target, newsamples_ref, newsamples_target, newpos
        
        
    return intro_array2, pop_df, intro_df



