# -*- coding: utf-8 -*-
import os
import re
import numpy as np
from scipy.stats import pearsonr, linregress

def extract_middle_float(line):
    parts = line.strip().split()
    return float(parts[2])

def detect_prefix(folder, dataset, classifier):
    files = os.listdir(folder)
    pattern = re.compile(r"{}_{}_([0-9]+)_\d+_evaluation\.txt".format(dataset, classifier))
    for fname in files:
        match = pattern.match(fname)
        if match:
            return "{}_{}_{}_".format(dataset, classifier, match.group(1))
    return None

def process_folder(base_path, dataset, classifier):
    clf_path = os.path.join(base_path, classifier)
    prefix = detect_prefix(clf_path, dataset, classifier)
    if not prefix:
        return None

    levels = [1] + list(range(10, 101, 10))
    files = ["{}/{}{:d}_evaluation.txt".format(clf_path, prefix, level) for level in levels]

    distances = []
    ifr_befores = []
    acc_befores = []
    ifr_afters = []
    acc_afters = []

    for filename in files:
        if not os.path.exists(filename):
            continue
        with open(filename, "r") as f:
            lines = f.readlines()
            acc_before = extract_middle_float(lines[1])
            ifr_before = extract_middle_float(lines[2])
            acc_after = extract_middle_float(lines[3])
            ifr_after = extract_middle_float(lines[4])
            distance = extract_middle_float(lines[5])

            distances.append(distance)
            acc_befores.append(acc_before)
            ifr_befores.append(ifr_before)
            acc_afters.append(acc_after)
            ifr_afters.append(ifr_after)

    if len(distances) < 2:
        return None

    acc_befores_pct = [x * 100 for x in acc_befores]
    acc_afters_pct = [x * 100 for x in acc_afters]
    ifr_befores_pct = [x * 100 for x in ifr_befores]
    ifr_afters_pct = [x * 100 for x in ifr_afters]

    ifr_improvements = []
    acc_improvements = []

    for i in range(len(acc_befores_pct)):
        b_ifr = ifr_befores_pct[i]
        a_ifr = ifr_afters_pct[i]
        b_acc = acc_befores_pct[i]
        a_acc = acc_afters_pct[i]

        imp_ifr = (b_ifr - a_ifr) / b_ifr * 100 if b_ifr != 0 else 0.0
        imp_acc = (a_acc - b_acc) / b_acc * 100 if b_acc != 0 else 0.0


        #if i == 10:
        #    print("[Level {}] Fairness: 100*({:.10f} - {:.10f}) / {:.10f} = {:.10f}".format(
        #        levels[i], b_ifr, a_ifr, b_ifr, imp_ifr))
        #    print("[Level {}] Accuracy: 100*({:.10f} - {:.10f}) / {:.10f} = {:.10f}".format(
        #        levels[i], a_acc, b_acc, b_acc, imp_acc))

        ifr_improvements.append(imp_ifr)
        acc_improvements.append(imp_acc)

    pcc_ifr, pval_ifr = pearsonr(distances, ifr_improvements)
    slope_ifr, _, _, _, _ = linregress(distances, ifr_improvements)
    pcc_acc, pval_acc = pearsonr(distances, acc_improvements)
    slope_acc, _, _, _, _ = linregress(distances, acc_improvements)

    pval_ifr_str = "%.1e" % pval_ifr
    pval_acc_str = "%.1e" % pval_acc

    output = []
    output.append("\\begin{tabular}{rr||rr|rr|rr}")
    output.append("\\multicolumn{4}{l}{4. %s-%s} & \\multicolumn{2}{c}{} & \\multicolumn{2}{c}{}\\\\" % (classifier, dataset.replace("_", "-")))
    output.append(" &  & \\multicolumn{2}{r|}{Before} & \\multicolumn{2}{r|}{After} & \\multicolumn{2}{r}{Impr. (\\%)}\\\\")
    output.append("Level & Div. & $\\mathrm{IFr}$ & Acc. & $\\mathrm{IFr}$ & Acc. & $\\mathrm{IFr}$ & Acc.\\\\")
    output.append("\\hline")

    avg_ifr_before = "%.2f" % np.mean(ifr_befores_pct)
    avg_acc_before = "%.2f" % np.mean(acc_befores_pct)

    for i in range(len(distances)):
        level = levels[i]
        div = round(distances[i], 2)
        ifr_b = "\\multirow{11}{*}{%s}" % avg_ifr_before if i == 0 else ""
        acc_b = "\\multirow{11}{*}{%s}" % avg_acc_before if i == 0 else ""
        ifr_a = "%.2f" % ifr_afters_pct[i]
        acc_a = "%.2f" % acc_afters_pct[i]
        ifr_imp = "%.2f" % ifr_improvements[i]
        acc_imp = "%.3f" % acc_improvements[i]
        output.append("%d & %.2f & %s & %s & %s & %s & %s & %s\\\\" % (
            level, div, ifr_b, acc_b, ifr_a, acc_a, ifr_imp, acc_imp))

    output.append("\\hline")
    output.append("\\multicolumn{2}{r}{Fairness ($\\mathrm{IFr}$)} & PCC: & \\multicolumn{1}{r}{%.2f} & p-value:  & \\multicolumn{1}{r}{%s} & Slope:  & %.2f\\\\" % (
        pcc_ifr, pval_ifr_str, slope_ifr))
    output.append("\\multicolumn{2}{r}{Accuracy} & PCC: & \\multicolumn{1}{r}{%.2f} & p-value:  & \\multicolumn{1}{r}{%s} & Slope:  & %.2f\\\\" % (
        pcc_acc, pval_acc_str, slope_acc))
    output.append("\\end{tabular}")

    return "\n".join(output)

datasets = ["BANK_age", "CENSUS_age", "CENSUS_race", "CENSUS_sex", "GERMAN_age", "GERMAN_sex"]
classifiers = ["SVM", "RF", "MLPC"]

for dataset in datasets:
    for clf in classifiers:
        tex_output = process_folder(dataset, dataset, clf)
        if tex_output:
            tex_dir = os.path.join(dataset, clf)
            if not os.path.exists(tex_dir):
                os.makedirs(tex_dir)
            filename = os.path.join(tex_dir, "{}_{}.tex".format(dataset, clf))
            with open(filename, "w") as f:
                f.write(tex_output)
            print("Saved:", filename)
