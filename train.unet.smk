import sys
sys.path.insert(0, './')

import numpy as np


## CONFIG


configfile: "config_intronets_archie1.yaml"

np.random.seed(config["seed"])

output_dir = config["output_dir"]
output_prefix = config["output_prefix"]
nrep = config["nrep"]
total_rep = config["total_rep"]
nrep_folder = int(total_rep / nrep)
nrep_folder_list = [x for x in range(nrep_folder)]
seed_list = np.random.random_integers(1, 2**31, nrep_folder)

### Config for rule simulating_training_data

demo_model_file = config["demes"]
nref = config["nref"]
ntgt= config["ntgt"]
ref_id = config["ref_id"]
tgt_id = config["tgt_id"]
src_id = config["src_id"]
seq_len = config["seq_len"]
mut_rate = config["mut_rate"]
rec_rate = config["rec_rate"]
ploidy = config["ploidy"]
is_phased = config["is_phased"]

### Config for rule create_h5_files

hdf_filename = config["hdf_filename"]

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


## RULES


rule all:
    input:
        output_dir + "/100k_random_wo_normal_net/best.weights",


rule simulate_training_data:
    input:
        demes = demo_model_file,
    output:
        flag = output_dir + "/100k_random_wo{nrep_folder}/.sim.completed",
    params:
        is_phased = '--phased' if is_phased else '',
        nrep = nrep,
        output_dir = output_dir + "/100k_random_wo{nrep_folder}",
        seed = lambda wildcards: seed_list[int(wildcards.nrep_folder)],
    resources:
        cpus = 8, partition="basic,himem",
    shell:
        """
        sstar simulate --demes {input.demes} --nref {nref} --ntgt {ntgt} --ref-id {ref_id} --tgt-id {tgt_id} --src-id {src_id} --mut-rate {mut_rate} --rec-rate {rec_rate} --seq-len {seq_len} --output-prefix {output_prefix} --output-dir {params.output_dir} --seed {params.seed} --replicate {params.nrep} --thread {resources.cpus} {params.is_phased} --keep-simulated-data
        touch {output.flag}
        """


rule create_h5_files:
    input:
        flags = expand(output_dir + "/100k_random_wo{nrep_folder}/.sim.completed", nrep_folder=nrep_folder_list)
    output:
        hdf_file = output_dir + "/100k_random_wo.h5",
        fwbw_hdf_file = output_dir + "/fwbw_100k_random_wo.h5",
        gradient_hdf_file = output_dir + "/gradient_100k_random_wo.h5",
        poschannel_hdf_file = output_dir + "/poschannel_100k_random_wo.h5",
        poschannel_scaled_hdf_file = output_dir + "/poschannel_scaled_100k_random_wo.h5",
    params:
        folders = expand(output_dir + "/100k_random_wo{nrep_folder}", nrep_folder=nrep_folder_list),
    resources:
        time = 180,
    run:
        import shutil
        from intronets_hdf import create_hdf_table_extrakey_chunk3_windowed, create_hdf_table_extrakey_chunk3_groups
        from intronets_hdf_extended import create_hdf_table_extrakey_chunk3_windowed_poschannel, create_hdf_table_extrakey_chunk3_windowed_gradient, create_hdf_table_extrakey_chunk3_windowed_forward_backward
        from intronets_process import process_vcf_df_multiproc
        from intronets_windows import process_vcf_df_windowed_multiproc

        gn = 0
        for f in params.folders:
            # PEP8: Don’t compare boolean values to True or False using ==
            # https://peps.python.org/pep-0008/
            if not no_window:
                all_entries = process_vcf_df_windowed_multiproc(f, polymorphisms=polymorphisms, stepsize=stepsize, 
                                                                random_reg=random_restrict, random_el=random_el, 
                                                                ignore_zero_introgression=remove_samples_wo_introgression, 
                                                                only_first=only_first)

                if create_extras:
                    create_hdf_table_extrakey_chunk3_windowed_poschannel(output.poschannel_hdf_file, all_entries, start_nr=gn)
                    create_hdf_table_extrakey_chunk3_windowed_poschannel(output.poschannel_scaled_hdf_file, all_entries, divide_by_seq_length=True, start_nr=gn)
                    create_hdf_table_extrakey_chunk3_windowed_gradient(output.gradient_hdf_file, all_entries, start_nr=gn)
                    create_hdf_table_extrakey_chunk3_windowed_forward_backward(output.fwbw_hdf_file, all_entries, start_nr=gn)

                gn = create_hdf_table_extrakey_chunk3_windowed(output.hdf_file, all_entries, start_nr=gn)
            else:
                all_entries = process_vcf_df_multiproc(f, polymorphisms=polymorphisms, 
                                                       remove_samples_wo_introgression=remove_samples_wo_introgression, 
                                                       random_restrict=random_restrict)
                gn = create_hdf_table_extrakey_chunk3_groups(output.hdf_file, all_entries, start_nr=gn)

            if return_data:
                collected_all_entries = []
                collected_all_entries.extend(all_entries)

            if remove_intermediate_data:
                shutil.rmtree(f)


rule train_unet_model:
    input:
        hdf_file = rules.create_h5_files.output.hdf_file,
    output:
        weights = output_dir + "/100k_random_wo_normal_net/best.weights",
    params:
        output_dir = output_dir + "/100k_random_wo_normal_net",
    resources:
        partition = "gpu",
        time = 1440,
    run:
        from intronets_train import train_model_intronets

        train_model_intronets(None, input.hdf_file, params.output_dir, net="default", n_classes=1, pickle_load=False, learning_rate = 0.001, batch_size=32, filter_multiplier=1, label_noise=0.01, n_early=10, label_smooth=True) 
