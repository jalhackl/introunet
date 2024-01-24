import numpy as np

configfile: "config/snakemake/config.yaml"

output_dir = config["output_dir"]
replicates = config["replicates"]
seq_len = config["seq_len"] * 10**6

np.random.seed(config["seed"])

seed_list = np.random.random_integers(1,2**31,replicates)

demo_models = config["demo_models"]
demo_model_ids = [x["id"] for x in demo_models]
mut_rates = [x["mut_rate"] for x in demo_models]
rec_rates = [x["rec_rate"] for x in demo_models]
ref_ids = [x["ref_id"] for x in demo_models]
tgt_ids = [x["tgt_id"] for x in demo_models]
src_ids = [x["src_id"] for x in demo_models]
demo_model_params = {}
for i in range(len(demo_model_ids)):
    demo_model_params[demo_model_ids[i]] = {}
    demo_model_params[demo_model_ids[i]]["mut_rate"] = mut_rates[i]
    demo_model_params[demo_model_ids[i]]["rec_rate"] = rec_rates[i]
    demo_model_params[demo_model_ids[i]]["ref_id"] = ref_ids[i]
    demo_model_params[demo_model_ids[i]]["tgt_id"] = tgt_ids[i]
    demo_model_params[demo_model_ids[i]]["src_id"] = src_ids[i]

nref_list = [50]
ntgt_list = [50]


rule all:
    input:
        expand(output_dir + "/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/0/sim.0.ts", 
               demog=demo_model_ids, nref=nref_list, ntgt=ntgt_list, seed=seed_list),
        expand(output_dir + "/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/0/sim.0.vcf", 
               demog=demo_model_ids, nref=nref_list, ntgt=ntgt_list, seed=seed_list),


rule simulation:
    input:
    output:
        ts = output_dir + "/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/0/sim.0.ts",
        vcf = output_dir + "/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/0/sim.0.vcf",
    params:
        demes = lambda wildcards: f'config/simulation/models/{wildcards.demog}.yaml',
        ref_id = lambda wildcards: demo_model_params[wildcards.demog]["ref_id"],
        tgt_id = lambda wildcards: demo_model_params[wildcards.demog]["tgt_id"],
        src_id = lambda wildcards: demo_model_params[wildcards.demog]["src_id"],
        mut_rate = lambda wildcards: demo_model_params[wildcards.demog]["mut_rate"],
        rec_rate = lambda wildcards: demo_model_params[wildcards.demog]["rec_rate"],
        seq_len = seq_len,
        output_prefix = "sim",
        output_dir = lambda wildcards: f'{output_dir}/simulated_data/{wildcards.demog}/nref_{wildcards.nref}/ntgt_{wildcards.ntgt}/{wildcards.seed}',
    resources:
        time=180, mem_mb=5000,
    shell:
        """
        sstar simulate --demes {params.demes} --nref {wildcards.nref} --ntgt {wildcards.ntgt} --ref-id {params.ref_id} --tgt-id {params.tgt_id} --src-id {params.src_id} --mut-rate {params.mut_rate} --rec-rate {params.rec_rate} --seq-len {params.seq_len} --output-prefix {params.output_prefix} --output-dir {params.output_dir} --seed {wildcards.seed}
        """
