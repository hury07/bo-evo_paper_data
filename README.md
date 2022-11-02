# Experiments on fast-hit
> All code and data used to run experiment on fast-hit in BO-EVO paper
## Requirements
 -fast-hit=0.0.0
## Code download
1. Clone this repository and `cd` into it.
    ```console
    git clone https://github.com/hury07/fasthit.git
    cd bo-evo_paper_data
    ```
## Run experiments
1. Run with empirical landscape or NK landscape.
    ~~~
    CUDA_VISIBLE_DEVICES=0 python code/run_main.py code/configs/gb1.budget.toml
    ~~~
    Configuration is set in .toml config file. A template is given in [code/configs/gb1.template.toml](code/configs/gb1.template.toml).
2. Run with wet-lab measurement.
    ~~~
    CUDA_VISIBLE_DEVICES=0 python code/run_rhla.py
    ~~~
    The process pauses between concecutive rounds, until the newly proposed variants are measured and the fitness data are updated to file [data/rhla/measured_seqs.csv](data/rhla/measured_seqs.csv).