from intronets_format import *
from intronets_hdf import *
from intronets_hdf_extended import *
from intronets_process import *
from intronets_windows import *


import os
import numpy as np
import sstar
#from sstar.simulate import simulate

#from sstar import __main__

import shutil

def create_hdf(total_rep, nrep, remove_intermediate_data=True, new_hdf_file="new_hdf.h5", demo_model_file="ArchIE_3D19.yaml", nref=50, ntgt=50, 
             ref_id='Ref', tgt_id='Tgt', src_id='Ghost', ploidy=2, seq_len=50000, mut_rate=1.25e-8, rec_rate=1e-8, thread=6,
             feature_config=None, is_phased=True, intro_prop=0.7, not_intro_prop=0.3, keep_sim_data=True,
             output_prefix='test', output_dir="test_output_dir", seed=None, create_extras=True, remove_samples_wo_introgression=False, random_restrict=True, random_el=1, one_target=True, polymorphisms=128, stepsize=16, only_first=True, no_window=False, return_data=False):
    
    if create_extras == True:
        poschannel_hdf_file = "poschannel_" + new_hdf_file
        gradient_hdf_file = "gradient_" + new_hdf_file
        fwbw_hdf_file = "fwbw_" + new_hdf_file

    nrep_folder = total_rep / nrep
    if nrep_folder < 1:
        nrep_folder = 1


    #if h5-file already exists, delete it
    if os.path.exists(new_hdf_file):
        os.remove(new_hdf_file)

    if create_extras == True:
        if os.path.exists(poschannel_hdf_file):
            os.remove(poschannel_hdf_file)
        if os.path.exists(gradient_hdf_file):
            os.remove(gradient_hdf_file)
        if os.path.exists(fwbw_hdf_file):
            os.remove(fwbw_hdf_file)



    if return_data == True:
        collected_all_entries = []
    #global collected_all_entries

    gn = 0
    for i in range(int(nrep_folder)):
        print("current counter i", i)
        new_output_dir = output_dir + str(i)

        '''
        sstar.simulate.simulate(demo_model_file=demo_model_file, nrep=nrep, nref=nref, ntgt=ntgt, 
                 ref_id=ref_id, tgt_id=tgt_id, src_id=src_id, ploidy=ploidy, seq_len=seq_len, mut_rate=mut_rate, rec_rate=rec_rate, thread=thread,
                 feature_config=None, is_phased=is_phased, intro_prop=intro_prop, not_intro_prop=not_intro_prop, keep_sim_data=True,
                 output_prefix=output_prefix, output_dir=new_output_dir, seed=None)


        _run_simulation(demo_model_file=demo_model_file, nrep=nrep, nref=nref, ntgt=ntgt,
            ref_id=ref_id, tgt_id=tgt_id, src_id=src_id, ploidy=ploidy, seq_len=seq_len, mut_rate=mut_rate, rec_rate=rec_rate, thread=thread,
            feature_config=None, is_phased=is_phased, intro_prop=intro_prop, not_intro_prop=not_intro_prop, keep_sim_data=True,
            output_prefix=output_prefix, output_dir=new_output_dir, seed=None)
        '''

        sstar_simulate_command = f"sstar simulate --demes {demo_model_file} --replicate {nrep} --nref {nref} --ntgt {ntgt} --ref-id {ref_id} --tgt-id {tgt_id} --src-id {src_id} --mut-rate {mut_rate} --rec-rate {rec_rate} --seq-len {seq_len} --output-prefix {output_prefix} --output-dir {new_output_dir} --thread {thread} --seed 12345 --keep-simulated-data --phased"

        newpid = os.fork()
        if newpid == 0:
            # we are in the child process
            os.execvp(sstar_simulate_command)
            os._exit(1)

        os.wait()
        #shell("sstar simulate --demes {demo_model_file} --replicate {nrep} --nref {nref} --ntgt {ntgt} --ref-id {ref_id} --tgt-id {tgt_id} --src-id {src_id} --mut-rate {mut_rate} --rec-rate #{rec_rate} --seq-len {seq_len} --output-prefix {output_prefix} --output-dir {new_output_dir} --thread 6 --seed 12345 --keep-simulated-data --phased")

        if no_window == False:
            all_entries = process_vcf_df_windowed_multiproc(new_output_dir, polymorphisms=polymorphisms, stepsize=stepsize, random_reg=random_restrict, random_el=random_el, ignore_zero_introgression=remove_samples_wo_introgression, only_first=only_first, one_target=one_target)
        else:
            all_entries = process_vcf_df_multiproc(new_output_dir, polymorphisms=polymorphisms, remove_samples_wo_introgression=remove_samples_wo_introgression, random_restrict=random_restrict, one_target=one_target)

        if return_data == True:
            collected_all_entries.extend(all_entries)

        if no_window == False:
            
            if create_extras == True:
                create_hdf_table_extrakey_chunk3_windowed_poschannel(poschannel_hdf_file, all_entries, start_nr=gn)
                create_hdf_table_extrakey_chunk3_windowed_gradient(gradient_hdf_file, all_entries, start_nr=gn)
                create_hdf_table_extrakey_chunk3_windowed_forward_backward(fwbw_hdf_file, all_entries, start_nr=gn)
            
            gn = create_hdf_table_extrakey_chunk3_windowed(new_hdf_file, all_entries, start_nr=gn)
        else:
            
            gn = create_hdf_table_extrakey_chunk3_groups(new_hdf_file, all_entries, start_nr=gn)



        if remove_intermediate_data == True:
            shutil.rmtree(new_output_dir)


    
    if return_data == True:
        return collected_all_entries
