# sstar2-analysis

There are **four snakemake files**:  



*intronets_simulate_training_set.snake.snake*: creates test files for training (default: 50kB); within a loop, simulation files are created using the sstar-functions, processed (e.g. seriation) and added to a h5-file



*intronets_simulate_prediction_set.snake*: creates test files for prediciton (default: 1Mb)

*intronets_train.snake*: training of the CNN

*intronets_infer.snake*: prediction using the CNN  

Furthermore, there are various additional scripts of the form *intronets_simulate_training_set_*; these snakemake-files create training data for the various models.
The *500k*-versions create samples until 500k windows with introgression are added to the h5-file.

the function parameters are given in 
config_intronets_archie1.yaml
config_intronets_archie1_predict.yaml
However, they are partially overwritten (in particular, the filenames) in the snakemake files


Functions from the following python functions are called:  



*intronets_format.py*: contains functions for loading and formatting vcf- and bed-(truth.tracts)-files

*intronets_seriation.py*: contains all functions necessary for seriation

*intronets_train.py*: pytorch training (modified version from the intronets-repository)

*intronets_infer.py*: pytorch prediction (heavily modified version from the intronets-repository)

*intronets_evaluate.py*: computation of precision-recall-curves according to a cutoff-list

*intronets_windows.py*: main functions for processing the vcf- and bed-inputfiles; the windows are created

*intronets_process.py*: this file contains main functions for processing the vcf- and bed-inputfiles without windowing (i.e., if only one window is processed); for the usual workflow, these files are not needed (however, for the training files, the position information is not necessary, so one could save space by using the non-windowed variant which does not store position information)

*intronets_hdf.py*: function for creating/extending the h5-file

*intronets_hdf_extended.py*: some variants of the h5-creation for network architectures with additional features (i.e. distances between SNPs)

*layers.py*: network layers from the intronets-repository with some additional architectures

*evaluate_unet_windowed_orig.py*: from the intronets-repository; only the function for creating the Gaussian smoothing is used

*data_loaders.py*: from the intronets-repository; only the h5-batch-loading function is used for training (in intronets_train.py)


