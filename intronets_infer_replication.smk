import sys
sys.path.insert(0, './')
import os

from intronets_infer import *


configfile: "config_intronets_archie1.yaml"

hdf_filename = config["hdf_filename_prediction"]
input_dir = config["output_dir_prediction"]
output_folder = config["prediction_dir"]
weightsfolder = training_dir
weights = os.path.join(weightsfolder, "best.weights")


rule all:
    resources:
        cpus = 2, partition="gpu",
        time = 400,
    run:
        predict_model_intronets(weights, os.path.join(input_dir, hdf_filename), output_folder, n_classes=1, smooth=False)