# Distributed deep learning with RAPIDS

This is a small repo with a demo of distributed training of a random forest using NVIDIA RAPIDS. 

It includes some example job submission scripts for different scheduling systems principally targeted at ARC4 and Bede.

## Usage

First create the conda environment (you will need to [install miniconda](https://docs.conda.io/en/latest/miniconda.html) if you haven't already)

```bash
$ conda env create -f environment.yml
```
Then check the submission script you wish to use to ensure it activates the correct conda environment with the name `rapids` using a line such as:

```bash
source activate rapids
```

Then submit your job using the appropriate job submission command:
```bash

# for SGE systems
$ qsub submit-sgeRapids.sh

# for slurm systems
$ sbatch submit-slurmRapids.sh

```

## TODO
- [ ] - Actually do some distributed deep learning. Following on the ideas in this [page from Keras docs](https://keras.io/guides/distributed_training/).
