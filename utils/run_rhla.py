import torch

import fasthit
from fasthit.utils import utils
import fasthit.utils.sequence_utils as s_utils

task = "rhla"
sequence_file = f"../data/{task}/proposed_seqs.csv"
fitness_file = f"../data/{task}/measured_seqs.csv"
wt_file = f"../data/{task}/wt_seq.fasta"
search_space = "R74,A101,L148,S173"
start_seq = "RALS"


def main():
    seed = 0
    rounds = 6
    expmt_queries_per_round = 384
    model_queries_per_round = 3200

    landscape = fasthit.landscapes.EXP(
        fitness_file, wt_file, search_space=search_space, sequence_csv=sequence_file
    )
    alphabet = s_utils.AAS

    utils.set_torch_seed(seed)
    encoder = fasthit.encoders.ESM(
        "esm-1v", alphabet,
        landscape.wt, landscape.combo_python_idxs,
        batch_size=32,
    )

    # utils.set_torch_seed(seed)
    model = fasthit.models.GPRegressor(kernel="RBF")

    log_file = (
        f"runs/{task}/run0.csv"
    )

    explorer = fasthit.explorers.bo.BO_EVO(
        encoder,
        model,
        rounds=rounds,
        expmt_queries_per_round=expmt_queries_per_round,
        model_queries_per_round=model_queries_per_round,
        starting_sequence=start_seq,
        alphabet=alphabet,
        log_file=log_file,
        util_func="UCB",
        uf_param=0.2,
    )

    explorer.run(landscape, verbose=True,
                 init_seqs_file=f"../data/{task}/init_seqs.csv")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
