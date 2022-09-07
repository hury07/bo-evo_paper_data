import pandas as pd
import fasthit

problem = fasthit.landscapes.gb1.registry()["with_imputed"]
landscape = fasthit.landscapes.GB1(**problem["params"])

proposed_seqs = pd.read_csv(
    "data/gb1/proposed_seqs.csv")["Variants"].to_numpy()
scores = landscape.get_fitness(proposed_seqs)

measured_seqs = pd.read_csv("data/gb1/measured_seqs.csv")
measured_seqs = measured_seqs.append(
    pd.DataFrame(
        {
            "Variants": proposed_seqs,
            "Fitness": scores,
        }
    )
)

measured_seqs.to_csv("data/gb1/measured_seqs.csv", index=False)
