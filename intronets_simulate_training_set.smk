import sys
sys.path.insert(0, './')


### CONFIG ###


configfile: "config_intronets_archie1.yaml"

output_dir = config["output_dir"]
output_prefix = config["output_prefix"]
nrep = config["nrep"]
total_rep = config["total_rep"]
nrep_folder = int(total_rep / nrep)

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

np.random.seed(config["seed"])
seed_list = np.random.random_integers(1, 2**31, nrep_folder)

nrep_folder_list = [x for x in range(nrep_folder)]


### RULES ###


rule all:
    input:
        output_dir + "/100k_random_wo.h5",
        output_dir + "/fwbw_100k_random_wo.h5",
        output_dir + "/gradient_100k_random_wo.h5",
        output_dir + "/poschannel_100k_random_wo.h5",
        output_dir + "/poschannel_scaled_100k_random_wo.h5",


rule simulate_training_data:
    input:
        demes = demo_model_file,
    output:
        flag = output_dir + "/100k_random_wo{nrep_folder}/.sim.completed",
    params:
        is_phased = '--phased' if is_phased is True else '',
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
        from intronets_hdf import *
        from intronets_hdf_extended import *
        from intronets_process import *
        from intronets_windows import *

        gn = 0
        for f in params.folders:
            if no_window is False:
                all_entries = process_vcf_df_windowed_multiproc(f, polymorphisms=polymorphisms, stepsize=stepsize, 
                                                                random_reg=random_restrict, random_el=random_el, 
                                                                ignore_zero_introgression=remove_samples_wo_introgression, 
                                                                only_first=only_first)

                if create_extras is True:
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

            if return_data == True:
                collected_all_entries = []
                collected_all_entries.extend(all_entries)

            if remove_intermediate_data == True:
                shutil.rmtree(f)
