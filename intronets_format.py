from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, cdist
from seriate import seriate
import numpy as np
import pandas as pd
import allel
import os
from pathlib import Path


def get_vcf_subfolders(folder):
    dir_list = os.listdir(folder)  
    for idir, directory in enumerate(dir_list):
        dir_list[idir] = os.path.join(folder,directory)
    return dir_list


def get_vcf_bed_files(folder, ignore_zero_introgression = True):
    
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
    
    dir_list = os.listdir(folder)    
    
    
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


def process_vcf_one_folder(folder):
    vcf_files, bed_files = get_vcf_bed_files(folder)
    
    haplos_list = []
    intros_list = []
    pos_list = []
    
    for vcf_file, bed_file in zip(vcf_files, bed_files):
        combined_haplos, combined_intros, newpos = format_from_vcf(vcf_file, bed_file)
        
        haplos_list.append(combined_haplos)
        intros_list.append(combined_intros)
        pos_list.append(newpos)
        
    return haplos_list, intros_list, pos_list



def process_vcf(folder):
    vcf_files, bed_files = get_vcf_bed_folder(folder)
    
    haplos_list = []
    intros_list = []
    pos_list = []
    
    for vcf_file, bed_file in zip(vcf_files, bed_files):
        combined_haplos, combined_intros, newpos = format_from_vcf(vcf_file, bed_file)
        
        haplos_list.append(combined_haplos)
        intros_list.append(combined_intros)
        pos_list.append(newpos)
        
    return haplos_list, intros_list, pos_list


def extract_genotype_matrix(vcf_file, merge_haplotypes=False):
    allel_vcf = allel.read_vcf(vcf_file)
    
    newar = allel_vcf["calldata/GT"]
    newpos = allel_vcf["variants/POS"]
    newsamples = allel_vcf["samples"]
    
    newar_haplo0 = newar[:,:,0]
    newar_haplo1 = newar[:,:,1]
    newar_haplo0T = newar_haplo0.T
    newar_haplo1T = newar_haplo1.T
    
    newar_haplo = np.array([newar_haplo0T, newar_haplo1T])
    
    return newpos, newsamples, newar_haplo



def format_from_vcf(vcf_file, bed_file, pop_size=None, turn_populations=True, merge_haplotypes=False, compute_scaled_positions=True, fragment_length=50000):
    allel_vcf = allel.read_vcf(vcf_file)
    
    newar = allel_vcf["calldata/GT"]
    newpos = allel_vcf["variants/POS"]
    newsamples = allel_vcf["samples"]
    

    newar_haplo0 = newar[:,:,0]
    newar_haplo1 = newar[:,:,1]
    newar_haplo0T = newar_haplo0.T
    newar_haplo1T = newar_haplo1.T
    
    
    newar_haplo = np.array([newar_haplo0T, newar_haplo1T])
    
    true_tract_data = pd.read_csv(bed_file, sep="\t", header=None, names=['chr', 'start', 'end', 'hap', 'ind'])

    true_tract_data["hap"] = true_tract_data["hap"].str.replace("hap_", "")
    true_tract_data["ind"] = true_tract_data["ind"].str.replace("tsk_", "")
    true_tract_data_arr = true_tract_data.to_numpy(dtype=int)
    
    
    intro_array = np.zeros_like(newar_haplo)
    for entry in true_tract_data_arr:

        curr_ind = entry[-1]
        curr_haplo = entry[-2]

        curr_start = entry[1]
        curr_end = entry[2]

        for ipos, pos in enumerate(newpos):
            if pos >= curr_start and pos < curr_end:

                intro_array[curr_haplo][curr_ind][ipos] = 1 
                
                
    combined_haplos = np.array([row for row_group in zip(newar_haplo[0], newar_haplo[1]) for row in row_group])
    combined_intros = np.array([row for row_group in zip(intro_array[0], intro_array[1]) for row in row_group])

    
    if turn_populations == True:
        combined_haplos_len = len(combined_haplos)
        
        if pop_size == None:
            half_len = int(combined_haplos_len / 2)
        else:
            half_len = int(combined_haplos_len - pop_size)

            
        combined_haplos_part0 = combined_haplos[0:half_len]
        combined_haplos_part1 = combined_haplos[half_len:]
        
        combined_intros_part0 = combined_intros[0:half_len]
        combined_intros_part1 = combined_intros[half_len:]
        
        new_samples_part0 = newsamples[0:int(half_len/2)]
        new_samples_part1 = newsamples[int(half_len/2):]

        combined_haplos = np.concatenate((combined_haplos_part1, combined_haplos_part0), axis=0)
        combined_intros = np.concatenate((combined_intros_part1, combined_intros_part0), axis=0)
        
        combined_samples = np.concatenate((new_samples_part1, new_samples_part0), axis=0)

    
    
    if compute_scaled_positions == True:
        scaled_pos = []
        for pos in newpos:
            scaled_pos.append(pos/fragment_length)
            
        return combined_haplos, combined_intros, scaled_pos
    
    
                    
    return combined_haplos, combined_intros, newpos




def format_from_vcf_df(vcf_file, bed_file, return_all=True, return_df=False, upsample=112/2, uniform_upsampling=True, ploidy=2, ref_size=50, target_size=50, select_populations=True, merge_haplotypes=False, compute_scaled_positions=True, fragment_length=50000, add_same_number=False):
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

    
    ref_size_full = ref_size * ploidy
    target_size_full = target_size * ploidy
    

    
    newsamples_int = [[int(s) for s in newsample.split("_") if s.isdigit()] for newsample in newsamples]
    newsamples_int = [item for sublist in newsamples_int for item in sublist]
    
    newsamples_ref = newsamples_int[0:ref_size]
    newsamples_target = newsamples_int[ref_size:]
    
    
    newar_haplo0 = newar[:,:,0]
    newar_haplo1 = newar[:,:,1]
    newar_haplo0T = newar_haplo0.T
    newar_haplo1T = newar_haplo1.T
    
    
    newar_haplo = np.array([newar_haplo0T, newar_haplo1T])

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
                
    
    if compute_scaled_positions == True:
        scaled_pos = []
        for pos in newpos:
            scaled_pos.append(pos/fragment_length)
            
    
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



