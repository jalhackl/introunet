import sys
sys.path.insert(0, './')

import numpy as np
import os
import shutil



## CONFIG


configfile: "config_intronets_archie_replication.yaml"

np.random.seed(config["seed"])

output_dir = config["output_dir_prediction"]
output_prefix = config["output_prefix"]
nrep = config["nrep_prediction"]
total_rep = config["total_rep_prediction"]

demo_model_file = config["demes"]
nref = config["nref"]
ntgt= config["ntgt"]
ref_id = config["ref_id"]
tgt_id = config["tgt_id"]
src_id = config["src_id"]
seq_len = config["seq_len_prediction"]
mut_rate = config["mut_rate"]
rec_rate = config["rec_rate"]
ploidy = config["ploidy"]
is_phased = config["is_phased"]


hdf_filename = config["hdf_filename_prediction"]
polymorphisms = config["polymorphisms"]
remove_samples_wo_introgression = config["remove_samples_wo_introgression"]
random_restrict = config["random_restrict"]
no_window = config["no_window"] #if no_window == True, no windowing is applied, but solely one random (or the first) window is chosen
stepsize = config["stepsize"]
random_el = config["random_el"]
only_first = config["only_first"] #if only_first, only the first window is used
return_data = config["return_data"]
create_extras = config["create_extras"] #if create_extras == True, also h5-files with additional information (position of SNPs, distances between adjacent SNPs, etc. are created)
remove_intermediate_data = config["remove_intermediate_data"]


inner_batch_size = config["inner_batch_size_prediction"] #inner batch size defines how many simulation folders are created for one iteration
#inner_batch_size should be a divisor of nrep (the number of simulations per iteration)
#each of the batches is processed by one process / cpu
#for prediction, (given that we predict longer chromosomes) a low inner_batch_size is preferable 
#in fact, for prediction (given that we predict longer and only a few chromosomes), one does not need batches, so one could use a simpler version (compare intronets_simulate_prediction_set.snake in scripts_for_testing_only) or also split simulaton and h5-creation
#however, for clarity it is perhaps better when the functions are analogous

## RULES

rule all:
    input:
        output_dataframe = os.path.join(output_dir, hdf_filename)


rule create_dataframes:
    output:
        output_dataframe = os.path.join(output_dir, hdf_filename),
    resources:
        cpus = 16, partition="basic",
        time = 2000,
    run:
        from intronets_hdf import create_hdf_table_extrakey_chunk3_windowed, create_hdf_table_extrakey_chunk3_groups
        from intronets_hdf_extended import create_hdf_table_extrakey_chunk3_windowed_poschannel, create_hdf_table_extrakey_chunk3_windowed_gradient, create_hdf_table_extrakey_chunk3_windowed_forward_backward
        from intronets_process import process_vcf_df_multiproc
        from intronets_windows import process_vcf_df_windowed_multiproc, process_simulations

        #nrep_folder indicates how many iterations are necessary (each having nrep simulations) to perform the desired number of simulations
        nrep_folder = int(total_rep / nrep)
        if nrep_folder < 1:
            nrep_folder = 1

        #ideally, the number of inner batches would be equal to the number of cpus*threads (but this probably depends on many factors, i.e. seq. length and total number of simulations)
        #in any case, it should be not smaller !

        #inner_nrep indicates how many subfolders (i.e. single simulations) are created within one folder (and each folder is processed by one process / cpu, i.e. they are processed in parallel)
        inner_nrep = int(max(nrep/inner_batch_size, 1))


        if return_data:
            collected_all_entries = []

        gn = 0
        for i in range(int(nrep_folder)):
            new_output_dir = output_dir + str(i)


            #process_simulation performs inner_nrep*inner_batch_size simulations
            #thread specifies how many processes should be allocated to one batch (i.e. to inner_batch_size simulations)
            new_output_dirs = process_simulations(new_output_dir, inner_batch_size=inner_batch_size, demo_model_file=demo_model_file, inner_nrep=inner_nrep, nref=nref, ntgt=ntgt, 
                        ref_id=ref_id, tgt_id=tgt_id, src_id=src_id, ploidy=ploidy, seq_len=seq_len, mut_rate=mut_rate, rec_rate=rec_rate, thread=2,is_phased=is_phased,
                        feature_config=None, 
                        output_prefix=output_prefix,  seed=None)

            #process simulation files (seriation etc.)
            #currently, we always use windows (i.e. no_window == False)
            if not no_window:
                all_entries = process_vcf_df_windowed_multiproc(new_output_dirs, polymorphisms=polymorphisms, stepsize=stepsize, random_reg=random_restrict, random_el=random_el, ignore_zero_introgression=remove_samples_wo_introgression, only_first=only_first, list_of_folders=True)
            else:
                all_entries = process_vcf_df_multiproc(new_output_dir, polymorphisms=polymorphisms, remove_samples_wo_introgression=remove_samples_wo_introgression, random_restrict=random_restrict)

            if return_data:
                collected_all_entries.extend(all_entries)

            #samples added to h5 dataframe
            if not no_window:
                gn = create_hdf_table_extrakey_chunk3_windowed(output.output_dataframe, all_entries, start_nr=gn)
            else:
                gn = create_hdf_table_extrakey_chunk3_groups(output.output_dataframe, all_entries, start_nr=gn)

            #before the next iteration starts, the simulation files have to be deleted
            if remove_intermediate_data:
                shutil.rmtree(new_output_dir)
