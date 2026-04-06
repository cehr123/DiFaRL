# DiFaRL

Welcome to the official DiFaRL repository!
Use these instructions to generate the results from figure 5.

## Blood Pressure Simulator

### Step 1: Generate MPD parameters / ground truth policy

`DiFaRL/BloodPressureSim/data-prep`

```
python 0-compute-MDP-parameters.py
python 1-ground-truth-policy.py
```

### Step 2: Generate Data

`DiFaRL/BloodPressureSim/data-prep`


```
python datagen.py --dir_data=<dir_data> --NSIMSAMPS=<NSIMSAMPS> --runs=<runs>
```

| Flag      |         Type | Description | Example                        |
| --------- | -----------: | ----------- | ------------------------------ |
| `--dir_data`     |        `str` | Epsilon value for the epsilon greedy data collection and corresponding saving directory. **Options:** eps_1, eps_0_5, eps_0_2, eps_0_1 | `--dir_data="eps_0_5"`                     |
| `--NSIMSAMPS`     |        `int` | Number of samples in each independent run. | `--NSIMSAMPS=1000`                     |
| `--runs`     |        `int` | Number of runs. | `--N=10`                     |

### Step 3: Run FQI

`DiFaRL/BloodPressureSim/exp-nets`

```
python run_FQI.py --N=<N> --run=<run> --dir="<dir>" --model=<model> --group=<group>
```

| Flag      |         Type | Description | Example                        |
| --------- | -----------: |  ----------- | ------------------------------ |
| `--N`     |        `int` | Number of samples. | `--N=100`                     |
| `--run`   |        `int` | Run index to produce multiple independent runs. | `--run=0`                      |
| `--dir`   |        `str` | Epsilon value for the corresponding input directory.  **Options:** eps_1, eps_0_5, eps_0_2, eps_0_1| `--dir="eps_0_5"` |
| `--model` |        `str` | Model architecture to use. **Options:** dense, attention| `--model="dense"`                  |
| `--group` |        `str` | Method for grouping together sub-actions. **Options:** baseline, factored, DiFaRL, oracle | `--group="baseline"`             |

To ensure that certain values of N, run, and dir are avaiable, be sure to generate the corresponding data using datagen. 

### Step 4: Run Evaluation

`DiFaRL/BloodPressureSim/exp-nets`

```
python run_eval_FQI.py --dir=<dir> --model=<model> --group=<group> --Ns=<Ns> --runs=<runs>
```

| Flag      |         Type | Description | Example                        |
| --------- | -----------: |  ----------- | ------------------------------ |
| `--dir`     |        `str` | Epsilon value of the input directory. **Options:** eps_1, eps_0_5, eps_0_2, eps_0_1 | `--dir="eps_0_5"` |
| `--model` |        `str` | Model architecture to use. **Options:** dense, attention| `--model="dense"`                  |
| `--group` |        `str` | Method for grouping together sub-actions. **Options:** baseline, factored, DiFaRL, oracle | 
| `--Ns`     |        `int` | Number of samples. | `--Ns 50 100 500 1000`                     |
| `--runs`     |        `int` | Number of samples. | `--runs=10`        

To ensure that certain values of N, run, and dir are avaiable, be sure to train the corresponding networks using run_FQI.py.        |
