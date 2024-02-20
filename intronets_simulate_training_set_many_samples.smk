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

#total rep indicates the total number of replicates!
total_rep =  100000
#nrep indicates the number of folders / simulations created per iteration
nrep =  1000
#hence nrep_folder indicates how many iterations have to be perform to obtain the desired number of simulations
nrep_folder = total_rep / nrep

#ideally, the number of inner batches would be equal to the number of cpus*threads (but this probably depends on many factors, i.e. seq. length and total number of simulations)
#in any case, it should be not smaller !

#the inner_batch_size indicates, how many simulations will be carried out in one subfolder 
inner_batch_size = int(max(nrep / 10, 1))
#inner_nrep indicates how many folders are created (and each folder is processed by one process / cpu, i.e. they are processed in parallel)
inner_nrep = int(max(nrep/inner_batch_size, 1))


rule all:
    run:
        if create_extras == True:
            poschannel_hdf_file = "poschannel_" + new_hdf_file
            poschannel_scaled_hdf_file = "poschannel_scaled_" + new_hdf_file

            gradient_hdf_file = "gradient_" + new_hdf_file
            fwbw_hdf_file = "fwbw_" + new_hdf_file

        #nrep_folder indicates how many iterations are necessary (each having nrep simulations) to perform the desired number of simulations
        nrep_folder = total_rep / nrep
        if nrep_folder < 1:
            nrep_folder = 1


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

        gn = 0
        for i in range(int(nrep_folder)):
            print("current counter i", i)
            new_output_dir = output_dir + str(i)


            #process_simulation performs inner_nrep*inner_batch_size simulations
            #thread specifies how many processes should be allocated to one batch (i.e. to inner_batch_size simulations)
            new_output_dirs = process_simulations(new_output_dir, inner_batch_size=inner_batch_size, demo_model_file=demo_model_file, inner_nrep=inner_nrep, nref=nref, ntgt=ntgt, 
                        ref_id=ref_id, tgt_id=tgt_id, src_id=src_id, ploidy=ploidy, seq_len=seq_len, mut_rate=mut_rate, rec_rate=rec_rate, thread=thread,is_phased=is_phased,
                        feature_config=None, 
                        output_prefix=output_prefix,  seed=None)
            
            print("simprocess accomplished")


            if no_window == False:
                all_entries = process_vcf_df_windowed_multiproc(new_output_dirs, polymorphisms=polymorphisms, stepsize=stepsize, random_reg=random_restrict, random_el=random_el, ignore_zero_introgression=remove_samples_wo_introgression, only_first=only_first, list_of_folders=True)
            else:
                all_entries = process_vcf_df_multiproc(new_output_dir, polymorphisms=polymorphisms, remove_samples_wo_introgression=remove_samples_wo_introgression, random_restrict=random_restrict)

            if return_data == True:
                collected_all_entries.extend(all_entries)

            #samples added to h5 dataframe
            if no_window == False:

                if create_extras == True:
                    create_hdf_table_extrakey_chunk3_windowed_poschannel(poschannel_hdf_file, all_entries, start_nr=gn)
                    create_hdf_table_extrakey_chunk3_windowed_poschannel(poschannel_scaled_hdf_file, all_entries, divide_by_seq_length=True, start_nr=gn)
                    create_hdf_table_extrakey_chunk3_windowed_gradient(gradient_hdf_file, all_entries, start_nr=gn)
                    create_hdf_table_extrakey_chunk3_windowed_forward_backward(fwbw_hdf_file, all_entries, start_nr=gn)

                gn = create_hdf_table_extrakey_chunk3_windowed(new_hdf_file, all_entries, start_nr=gn)
            else:

                gn = create_hdf_table_extrakey_chunk3_groups(new_hdf_file, all_entries, start_nr=gn)


            #before the next iteration starts, the simulation files have to be deleted
            if remove_intermediate_data == True:
                shutil.rmtree(new_output_dir)
