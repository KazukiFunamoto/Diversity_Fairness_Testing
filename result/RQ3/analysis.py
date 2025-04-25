# -*- coding: utf-8 -*-
import os
import re

def extract_middle_value(line):
    # Extract float values including scientific notation
    numbers = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', line)
    if len(numbers) >= 2:
        return float(numbers[1])
    return None

def compute_improvements(before_acc, after_acc, before_fair, after_fair):
    if None in [before_acc, after_acc, before_fair, after_fair]:
        return None, None
    acc_imp = 100.0 * (after_acc - before_acc) / before_acc if before_acc != 0 else 0.0
    fair_imp = 100.0 * (before_fair - after_fair) / before_fair if before_fair != 0 else 0.0
    return acc_imp, fair_imp

def process_folder(folder_name):
    acc_list = []
    fair_list = []
    for fname in os.listdir(folder_name):
        if fname.endswith("_evaluation.txt"):
            path = os.path.join(folder_name, fname)
            with open(path, "r") as f:
                lines = f.readlines()
                before_acc = after_acc = before_fair = after_fair = None
                for line in lines:
                    if "before_accuracy_Data" in line:
                        before_acc = extract_middle_value(line)
                    elif "after_accuracy_Data" in line:
                        after_acc = extract_middle_value(line)
                    elif "before_fairness_Data" in line:
                        before_fair = extract_middle_value(line)
                    elif "after_fairness_Data" in line:
                        after_fair = extract_middle_value(line)
                acc_imp, fair_imp = compute_improvements(before_acc, after_acc, before_fair, after_fair)
                if acc_imp is not None and fair_imp is not None:
                    acc_list.append((fname, acc_imp))
                    fair_list.append((fname, fair_imp))

    # Output results to a text file at the same level as analysis.py
    if acc_list:
        out_path = folder_name + "_summary.txt"
        with open(out_path, "w") as out:
            for i in range(len(acc_list)):
                out.write("Accuracy Improvement: {:.3f}, Fairness Improvement: {:.3f} ({})\n".format(
                    acc_list[i][1], fair_list[i][1], acc_list[i][0]))
            out.write("\n")
            acc_avg = sum([x[1] for x in acc_list]) / len(acc_list)
            fair_avg = sum([x[1] for x in fair_list]) / len(fair_list)
            out.write("Average Accuracy Improvement: {:.3f}\n".format(acc_avg))
            out.write("Average Fairness Improvement: {:.3f}\n".format(fair_avg))
        print("Summary written to: {}".format(out_path))
    else:
        print("No valid files found in folder: {}".format(folder_name))

if __name__ == "__main__":
    target_folders = [
        "const_const", "const_majority", "const_random",
        "pair_const", "pair_majority", "pair_random",
        "random_const", "random_random", "random_majority"
    ]
    for folder in target_folders:
        if os.path.isdir(folder):
            process_folder(folder)
        else:
            print("Folder does not exist:", folder)

