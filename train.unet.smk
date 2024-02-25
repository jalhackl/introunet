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
demog_id = config["demog_id"]
ref_id = config["ref_id"]
tgt_id = config["tgt_id"]
src_id = config["src_id"]
nref = config["nref"]
ntgt= config["ntgt"]
seq_len = config["seq_len"]
mut_rate = config["mut_rate"]
rec_rate = config["rec_rate"]
ploidy = config["ploidy"]
is_phased = config["is_phased"]

### Config for rule create_h5_files

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
batch_size_list = config["batch_sizes"]


## RULES


rule all:
    input:
        expand(output_dir + "/{demog_id}/{output_prefix}_normal_net/batch_size_{batch_size}/best.weights",
               demog_id=demog_id, output_prefix=output_prefix, batch_size=batch_size_list)


rule simulate_training_data:
    input:
        demes = demo_model_file,
    output:
        flag = output_dir + "/{demog_id}/{output_prefix}_{nrep_folder}/.sim.completed",
    params:
        is_phased = '--phased' if is_phased else '',
        nrep = nrep,
        output_dir = output_dir + "/{demog_id}/{output_prefix}_{nrep_folder}",
        seed = lambda wildcards: seed_list[int(wildcards.nrep_folder)],
    resources:
        cpus = 8, partition="basic",
    shell:
        """
        sstar simulate --demes {input.demes} --nref {nref} --ntgt {ntgt} --ref-id {ref_id} --tgt-id {tgt_id} --src-id {src_id} --mut-rate {mut_rate} --rec-rate {rec_rate} --seq-len {seq_len} --output-prefix {output_prefix} --output-dir {params.output_dir} --seed {params.seed} --replicate {params.nrep} --thread {resources.cpus} {params.is_phased} --keep-simulated-data
        touch {output.flag}
        """


rule create_h5_files:
    input:
        lags = expand(output_dir + "/{demog_id}/{output_prefix}_{nrep_folder}/.sim.completed", nrep_folder=nrep_folder_list, allow_missing=True)
    output:
        hdf_file = output_dir + "/{demog_id}/{output_prefix}.h5",
        fwbw_hdf_file = output_dir + "/{demog_id}/fwbw_{output_prefix}.h5",
        gradient_hdf_file = output_dir + "/{demog_id}/gradient_{output_prefix}.h5",
        poschannel_hdf_file = output_dir + "/{demog_id}/poschannel_{output_prefix}.h5",
        poschannel_scaled_hdf_file = output_dir + "/{demog_id}/poschannel_scaled_{output_prefix}.h5",
    params:
        folders = expand(output_dir + "/{demog_id}/{output_prefix}_{nrep_folder}", nrep_folder=nrep_folder_list, allow_missing=True),
    resources:
        time = 1440,
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
        weights = output_dir + "/{demog_id}/{output_prefix}_normal_net/batch_size_{batch_size}/best.weights",
    params:
        output_dir = output_dir + "/{demog_id}/{output_prefix}_normal_net/batch_size_{batch_size}",
    benchmark:
        repeat("benchmarks/{demog_id}/{output_prefix}_normal_net_batch_size_{batch_size}.benchmark.txt", 3)
    resources:
        partition = "gpu",
        time = 1440,
        gres = "--gres=gpu:a30:1",
    run:
        from intronets_train import train_model_intronets

        train_model_intronets(None, input.hdf_file, params.output_dir, net="default", n_classes=1, pickle_load=False, learning_rate = 0.001, batch_size=int(wildcards.batch_size), filter_multiplier=1, label_noise=0.01, n_early=10, label_smooth=True) 
