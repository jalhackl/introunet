import sys
sys.path.insert(0, './')


from intronets_train import train_model_intronets


prefixname = "results/training_data/100k_random_wo"
hdf_filename = prefixname +".h5"
output_dir = prefixname + "_normal_net"


rule all:
    run:
        train_model_intronets(None, hdf_filename, output_dir, net="default", n_classes=1, pickle_load=False, learning_rate = 0.001, batch_size=32, filter_multiplier=1, label_noise=0.01, n_early=10, label_smooth=True)

