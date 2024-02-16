#!/bin/bash
#SBATCH --job-name=run-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2gb
#SBATCH --time=24:00:00
#SBATCH --exclude=node-a02,node-a08,node-a09,node-a11,node-a19,node-a24,node-a26,node-a27,node-a28,node-a29

snakemake -s intronets_simulate_training_set.smk --profile config/slurm
