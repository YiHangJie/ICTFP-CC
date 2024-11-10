import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from run_exp import Experiments
from utils.excel import save_to_excel

# 添加命令行参数设置，包括数据集，方法，攻击强度
# 例如 python main.py --dataset Benchmark --method CCA --attack 0.1
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Benchmark", choices=["Benchmark", "BETA"], help="Dataset to use")
parser.add_argument("--method", type=str, default="CCA", choices=["CCA", "FBCCA", "MSI"], help="Method to use")
parser.add_argument("--attack", type=float, default=0.1, help="Attack strength")
args = parser.parse_args()

#将命令行参数传递给变量
datasets = [args.dataset]
methods = [args.method]
attack = args.attack
subjects = 35 if args.dataset == "Benchmark" else 70
subjects = 1

# wss = np.round(np.linspace(1, 3, 11), 1) if datasets=="Benchmark" else np.round(np.linspace(1, 2.4, 8), 1)
wss = [2]

for dataset in datasets:
    for method in methods:
        print(f"doing experiment on {dataset} using {method}")
        
        headline = ["被试"]
        for ws in wss:
            headline += [f"raw_ACC_{ws}s", f"raw_ITR_{ws}s", f"attacked_ACC_{ws}s", f"attacked_ITR_{ws}s", f"defense_ACC_{ws}s", f"defense_ITR_{ws}s", 
                         f"TotalVarMin_ACC_{ws}s", f"TotalVarMin_ITR_{ws}s", 
                         f"Resample_ACC_{ws}s", f"Resample_ITR_{ws}s", 
                         f"FeatureSqueezing_ACC_{ws}s", f"FeatureSqueezing_ITR_{ws}s", 
                         f"SpatialSmoothing_ACC_{ws}s", f"SpatialSmoothing_ITR_{ws}s", ]

        res = []
        # run exp and get results

        for subject in range(1, subjects + 1):
            print(f"getting S{subject}'s results")
            sub_res = [f"S{subject}"]
            for ws in wss:
                exp1 = Experiments(dataset, attack=0, ws=ws, subjects=[subject], method=method)
                raw_acc, raw_ITR = exp1.run()
                exp2 = Experiments(dataset, attack=attack, ws=ws, subjects=[subject], method=method)
                attacked_acc, attacked_ITR = exp2.run()
                exp4 = Experiments(dataset, attack=attack, ws=ws, subjects=[subject], method=method, baseline="TotalVarMin", )
                TotalVarMin_acc, TotalVarMin_ITR = exp4.run_baseline()
                exp5 = Experiments(dataset, attack=attack, ws=ws, subjects=[subject], method=method, baseline="Resample")
                Resample_acc, Resample_ITR = exp5.run_baseline()
                exp6 = Experiments(dataset, attack=attack, ws=ws, subjects=[subject], method=method, baseline="FeatureSqueezing")
                FeatureSqueezing_acc, FeatureSqueezing_ITR = exp6.run_baseline()
                exp7 = Experiments(dataset, attack=attack, ws=ws, subjects=[subject], method=method, baseline="SpatialSmoothing")
                SpatialSmoothing_acc, SpatialSmoothing_ITR = exp7.run_baseline()

                sub_res += [raw_acc, raw_ITR, attacked_acc, attacked_ITR, 0, 0,
                            TotalVarMin_acc, TotalVarMin_ITR, 
                            Resample_acc, Resample_ITR, 
                            FeatureSqueezing_acc, FeatureSqueezing_ITR, 
                            SpatialSmoothing_acc, SpatialSmoothing_ITR,]
            res.append(sub_res)

        # calculate subject average results
        res.append(["average"] + [np.mean(np.array(res)[:, i].astype(float)) for i in range(1, len(headline))])

        df = pd.DataFrame(data=res, columns=headline)
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"result.xlsx")
        sheet_name = f"{dataset}_{method}_{attack}"
        save_to_excel(df, file_path, sheet_name)


        
# conda activate torch
# python ./Experiments/Defense_Performance_Comparison_baseline/main.py --dataset BETA --method MSI --attack 0.3