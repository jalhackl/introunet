import sys
sys.path.insert(0, './')

import sstar
import os
import shutil
import demes
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import nbinom

from intronets_format import *
from intronets_hdf import *
from intronets_hdf_extended import *
from intronets_process import *
from intronets_windows import *

from sstar.simulate import simulate


configfile: "config_intronets_archie1.yaml"
model_name = "archie"
output_dir = config["output_dir"]
demo_model_file = config["simulation_yamls"]["archie"]

nrep = 1000
nref = config["nref"]
ntgt= config["ntgt"]
ref_id = config["ref_ids"]["archie"]
tgt_id = config["tgt_ids"]["archie"]
src_id = config["src_ids"]["archie"]
seq_len = config["seq_len"]
mut_rate = config["mutation_rates"]["archie"]
rec_rate = config["recombination_rates"]["archie"]
output_prefix = "archie_rep_model"
output_dir = config["output_dir"]
seed = config["seed"]

hdf_filename = config["hdf_filename"]

hdf_filename = "100k_random_wo.h5"
new_hdf_file = hdf_filename
output_dir = "100k_random_wo"

ploidy = 2

#thread indicates how many processes are used for ONE BATCH (and one iteration consists of inner_nrep batches)
thread = 2

is_phased = True

#if create_extras == True, also h5-files with additional information (position of SNPs, distances between adjacent SNPs, etc. are created)
create_extras = True
#if no_window == True, no windowing is applied, but solely one random (or the first) window is chosen
no_window=False

remove_intermediate_data=True
remove_samples_wo_introgression=True
random_restrict=True
random_el=1
polymorphisms=128
stepsize=16
#if only_first, only the first window is used
only_first=True
return_data = False


#ideally, the number of inner batches would be equal to the number of cpus*threads
#in any case, it should be not smaller !
#inner_batch_size indicates how many subfolders are created within one batch
inner_batch_size = 100
#inner_nr indicates how many folders are created during one iteration
inner_nrep = 100
#100*100=10000 simulations are carried out per iteration

#how many samples do we need for the training set
#depending on the model 500k simulations with introgression mean that much more have to be carried out (e.g. for the archie-model approx. 1.3 mio simulations will be necessary)
desired_samples = 500000

#for the values in polymorphisms_list h5-dataframes are created
#we will probably stick to 192 (after the first results which show that 192 is significantly better than 128), but for different architectures / models perhaps windows of different size could be advantageous
polymorphisms_list = [128,192]


rule all:
    run:
        #variable counting the number of created samples
        created_samples = 0

        #filenames for creation of 'extra'-h5 files (e.g. with positional information)
        if create_extras == True:
            poschannel_hdf_file = "poschannel_" + new_hdf_file
            poschannel_scaled_hdf_file = "poschannel_scaled_" + new_hdf_file

            gradient_hdf_file = "gradient_" + new_hdf_file
            fwbw_hdf_file = "fwbw_" + new_hdf_file



        #if h5-file already exists, delete it
        if os.path.exists(new_hdf_file):
            os.remove(new_hdf_file)

        if create_extras == True:
            if os.path.exists(poschannel_hdf_file):
                os.remove(poschannel_hdf_file)
            if os.path.exists(poschannel_scaled_hdf_file):
                os.remove(poschannel_hdf_file)
            if os.path.exists(gradient_hdf_file):
                os.remove(gradient_hdf_file)
            if os.path.exists(fwbw_hdf_file):
                os.remove(fwbw_hdf_file)


        
        if return_data == True:
            collected_all_entries = []

        #gn counts the replicate number for the h5-dataframe
        gn = 0

        #additional iteration counter
        i = 0

        #iteration until enough samples are collected
        while (created_samples < desired_samples):
            print("current counter i", i)
            new_output_dir = output_dir + str(i)

            i = i + 1

            #process_simulation performs inner_nrep*inner_batch_size simulations
            #thread specifies how many processes should be allocated to one batch (i.e. to inner_batch_size simulations)

            new_output_dirs = process_simulations(new_output_dir, inner_batch_size=inner_batch_size, demo_model_file=demo_model_file, inner_nrep=inner_nrep, nref=nref, ntgt=ntgt, 
                        ref_id=ref_id, tgt_id=tgt_id, src_id=src_id, ploidy=ploidy, seq_len=seq_len, mut_rate=mut_rate, rec_rate=rec_rate, thread=thread,is_phased=is_phased,
                        feature_config=None, 
                        output_prefix=output_prefix,  seed=None)
            
            print("simprocess accomplished")

            #for all polymorphism numbers specified in polymorphisms_list a h5-dataframe is created
            for ipoly, polymorphisms in enumerate(polymorphisms_list):
                poly_prefix = str(polymorphisms) + "_"
                if no_window == False:
                    all_entries = process_vcf_df_windowed_multiproc(new_output_dir, polymorphisms=polymorphisms, stepsize=stepsize, random_reg=random_restrict, random_el=random_el, ignore_zero_introgression=remove_samples_wo_introgression, only_first=only_first)
                else:
                    all_entries = process_vcf_df_multiproc(new_output_dir, polymorphisms=polymorphisms, remove_samples_wo_introgression=remove_samples_wo_introgression, random_restrict=random_restrict)

                #counter is increased
                if ipoly == 0:
                    created_samples = created_samples + len(all_entries) 

                if return_data == True:
                    collected_all_entries.extend(all_entries)

                #samples added to h5 dataframe
                if no_window == False:

                    if create_extras == True:
                        create_hdf_table_extrakey_chunk3_windowed_poschannel(poly_prefix + poschannel_hdf_file, all_entries, start_nr=gnlist[ipoly])
                        create_hdf_table_extrakey_chunk3_windowed_poschannel(poly_prefix + poschannel_scaled_hdf_file, all_entries, divide_by_seq_length=True, start_nr=gnlist[ipoly])
                        create_hdf_table_extrakey_chunk3_windowed_gradient(poly_prefix + gradient_hdf_file, all_entries, start_nr=gnlist[ipoly])
                        create_hdf_table_extrakey_chunk3_windowed_forward_backward(poly_prefix + fwbw_hdf_file, all_entries, start_nr=gnlist[ipoly])

                    gnlist[ipoly] = create_hdf_table_extrakey_chunk3_windowed(poly_prefix + new_hdf_file, all_entries, start_nr=gnlist[ipoly])
                else:

                    gnlist[ipoly] = create_hdf_table_extrakey_chunk3_groups(poly_prefix + new_hdf_file, all_entries, start_nr=gnlist[ipoly])


            #before the next iteration starts, the simulation files have to be deleted
            if remove_intermediate_data == True:
                shutil.rmtree(new_output_dir)
