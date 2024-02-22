import sys
sys.path.insert(0, './')


from intronets_train import *



configfile: "config_intronets_archie1.yaml"

model_name = "archie"

output_dir = config["output_dir"]


demo_model_file = config["demes"]

nrep = 1000
nref = 50
ntgt= 50
ref_id = config["ref_id"]
tgt_id = config["tgt_id"]
src_id = config["src_id"]
seq_len = config["seq_len"]
mut_rate = config["mut_rate"]
rec_rate = config["rec_rate"]
thread = 32
output_prefix = "archie_rep_model"
output_dir = config["output_dir"]
seed = config["seed"]

hdf_filename = config["hdf_filename"]
#total rep indicates the total number of replicates!!!
total_rep =  config["total_rep"]
nrep =  config["nrep"]

prefixname = "results/training_data/1000k_random_wo"
hdf_filename = prefixname +".h5"
output_dir = prefixname + "normal_net"


rule all:
    run:
        train_model_intronets(None, hdf_filename, output_dir, net="default", n_classes=1, pickle_load=False, learning_rate = 0.001, batch_size=32, filter_multiplier=1, label_noise=0.01, n_early=10, label_smooth=True)

