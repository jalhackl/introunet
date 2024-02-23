import sys
sys.path.insert(0, './')


from intronets_train import *

configfile: "config_intronets_archie_replication.yaml"

model_name = "archie"
hdf_filename = config["hdf_filename"]
output_dir = config["training_dir"]

rule all:
    run:
        train_model_intronets(None, hdf_filename, output_dir, net="default", n_classes=1, pickle_load=False, learning_rate = 0.001, batch_size=32, filter_multiplier=1, label_noise=0.01, n_early=10, label_smooth=True)