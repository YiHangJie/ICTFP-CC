# ICTFP-CC

This repository contains the code for the paper: Independent Components Time-Frequency Purification with Channel Consensus Against Adversarial Attack in SSVEP-based BCIs. 

The code is written in Python 3.11. To run the code, you need to install the packages by running `pip install -r requirements.txt`.

## Running the code

To run the code, you need to follow these steps:

1. Download the Benchmark and BETA dataset from https://bci.med.tsinghua.edu.cn/download.html.
2. Extract every data files to the `data/Benchmark` folder or `data/BETA` folder.
3. Run the `data/Benchmark_read_attack.py` file and `data/BETA_read_attack.py` file to generate the attack data.
4. Run the `Experiments/Defense_Performance_Comparison_baseline/main.py` file and `Experiments/Defense_Performance_Comparison_ICTFP-CC/main.py` file to compare the performance of the baseline and ICTFP-CC defense methods.
5. Run the `statistically_independent.ipynb` file to generate the statistical independence results.

## Acknowledgement
The baseline algorithm was adapted from the open-source library [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox), and we would like to express our gratitude for their contribution.
