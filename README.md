# sstar2-analysis


In this branch, there are all snakemake-files needed to reproduce the results for the original introUnet-architecture.

For this purpose **four snakemake files are necessary**:

*intronets_simulate_training_replication.smk*: this snake-make creates the h5-file necessary for training (1Mb samples of 50 kB length are simulated, only those with introgression are processed)

*intronets_train_replication.smk*: training of the pytorch-model

*intronets_simulate_prediction_replication.smk*: this snake-make creates the h5-file necessary for prediction (1000 samples of 1Mb length are simulated)

*intronets_infer_replication.smk*: This snakemake-file does inference, precision-recall curves are created

The parameters are given in 
*config_intronets_archie_replication.yaml*



----------------------------

In the folders 'scripts_for_testing_only', there are **additional snakemake files**;
in particular, *intronets_simulate_training_set_many_samples.smk* and *intronets_simulate_training_set_fixed_sample_size.smk*.

For the replication of the introUnet-results, these files can be ignored; they are solely for testing purposes.
However, *intronets_simulate_training_set_many_samples.smk* implements the structure necessary for the further steps: The simulation process is iterated until enough introgressed samples have been obtained.


Furthermore, there are various additional scripts of the form *intronets_simulate_training_set_* in the folder 'additional_train_infer_scripts'; these snakemake-files create training data for the various models. They are for testing purposes and can be ignored.
The *500k*-versions create samples until 500k windows with introgression are added to the h5-file.




----------------------------


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


