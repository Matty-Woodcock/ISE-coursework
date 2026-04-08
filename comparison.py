import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

#Before running this file, make sure the required CSV files are in baseline_results and solution_results
#If you are missing any CSV files, please run solution.py or modified_br_classification.py to generate new CSV files
#Run this file for full comparison between my solution and base line, including satistical significance testing

PROJECTS = ["pytorch", "tensorflow", "keras", "incubator-mxnet", "caffe"]
#PROJECTS = ["keras"]
METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

for project in PROJECTS:
    baseline_pd = pd.read_csv(f"baseline_results/{project}_NB.csv")
    solution_pd = pd.read_csv(f"solution_results/{project}.csv")

    print(f"Project: {project}\n")

    for metric in METRICS:
        baseline_single_metric_results = baseline_pd[metric].values
        solution_single_metric_results = solution_pd[metric].values

        statistic, p_value = mannwhitneyu(solution_single_metric_results, baseline_single_metric_results, alternative="two-sided")

        print(f"{metric}")
        print(f"Baseline Median: {np.median(baseline_single_metric_results):.4f}")
        print(f"Solution Median: {np.median(solution_single_metric_results):.4f}")
        print(f"p-value: {p_value:.2g}")
        if p_value < 0.05:
            print("Satistically Signicant, Reject H0\n")
        else:
            print("Not Satistically Signicant, Cannot Reject H0\n")

#TO DO: Investigate keras behaviour further