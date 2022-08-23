# Adversarial Arena
Comparison of several adversarial attacks

Note that for wasserstein attack, the code is implemented in [this](https://github.com/bezirganyan/fast-wasserstein-adversarial) fork, a pull request is currently made for merging with the original repo. 

## Installation

Installation of the project (together with supporting repositories) is in `/scripts/setup_arena.sh`. 
To install simply run it as

```
./scripts/setup_arena.sh
```

The script
* Downloads necessary repositories for all experiments
* Downloads and unpacks the ImageNette dataset
* Creates a conda environment for the experiments
* Installs the necessary dependencies for the experiments

