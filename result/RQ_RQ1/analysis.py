import os
import re
import numpy as np
from scipy.stats import pearsonr

CORR_ACC_FILE = "correlation_accuracy_vs_diversity.txt.txt"
CORR_FAIR_FILE = "correlation_fairness_vs_diversity.txt"
REGRESSION_ACC_FILE = "regression_accuracy_vs_diversity.txt"
REGRESSION_FAIR_FILE = "regression_fairness_vs_diversity.txt"

# Extract the second float value from a text line
def extract_middle_float(line):
    return float(re.findall(r"[-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?", line)[1])

# Parse evaluation file and return (accuracy improvement, fairness improvement, distance)
def load_evaluation(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    before_acc = extract_middle_float(lines[1])
    after_acc = extract_middle_float(lines[3])
    acc_improve = (100.0 * after_acc - 100.0 * before_acc) / before_acc if before_acc != 0 else 0.0

    before_disc = extract_middle_float(lines[2])
    after_disc = extract_middle_float(lines[4])
    disc_improve = (100.0 * before_disc - 100.0 * after_disc) / before_disc if before_disc != 0 else 0.0

    distance = extract_middle_float(lines[5])
    return acc_improve, disc_improve, distance

# Perform correlation and regression analysis
def run_correlation():
    acc_results = []
    fair_results = []
    regression_acc_lines = []
    regression_fair_lines = []

    for root, dirs, files in os.walk(os.getcwd()):
        evaluation_files = [f for f in files if f.endswith("_evaluation.txt")]
        if not evaluation_files:
            continue

        acc_improvements = []
        fair_improvements = []
        distances = []

        for filename in evaluation_files:
            filepath = os.path.join(root, filename)
            try:
                acc_imp, fair_imp, dist = load_evaluation(filepath)
                acc_improvements.append(acc_imp)
                fair_improvements.append(fair_imp)
                distances.append(dist)
            except Exception as e:
                print("Error processing %s: %s" % (filepath, str(e)))

        if len(acc_improvements) >= 2:
            acc_corr, acc_p = pearsonr(acc_improvements, distances)
            fair_corr, fair_p = pearsonr(fair_improvements, distances)

            folder_label = os.path.basename(os.path.dirname(filepath))
            parent_folder = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
            label = "%s_%s" % (parent_folder, folder_label)

            acc_results.append((acc_corr, acc_p, label))
            fair_results.append((fair_corr, fair_p, label))

            # Linear regression: y = slope * x + intercept
            try:
                acc_coef = np.polyfit(distances, acc_improvements, 1)
                fair_coef = np.polyfit(distances, fair_improvements, 1)
                regression_acc_lines.append((label, acc_coef[0], acc_coef[1], acc_corr, acc_p))
                regression_fair_lines.append((label, fair_coef[0], fair_coef[1], fair_corr, fair_p))
            except Exception as e:
                print("Regression error for %s: %s" % (label, str(e)))

    def write_corr(outfile, results):
        with open(outfile, "w") as f:
            for corr, pval, label in results:
                f.write("correlation:%.3f, p-value:%.2e (%s)\n" % (corr, pval, label))

            f.write("\n")
            pos = [(label, corr) for corr, pval, label in results if pval < 0.05 and corr > 0]
            neg = [(label, corr) for corr, pval, label in results if pval < 0.05 and corr < 0]

            if pos:
                mean_pos = np.mean([c for _, c in pos])
                f.write("positive correlation (p < 0.05): mean = %.3f\n" % mean_pos)
                for label, corr in pos:
                    f.write("  - %s: %.3f\n" % (label, corr))

            if neg:
                f.write("\n")
                mean_neg = np.mean([c for _, c in neg])
                f.write("negative correlation (p < 0.05): mean = %.3f\n" % mean_neg)
                for label, corr in neg:
                    f.write("  - %s: %.3f\n" % (label, corr))

    def write_regression(outfile, lines):
        with open(outfile, "w") as f:
            for label, slope, intercept, corr, pval in lines:
                f.write("y = %.3f * x + %.3f (%s)\n" % (slope, intercept, label))

            f.write("\n")
            all_slopes = [slope for _, slope, _, _, _ in lines]
            f.write("all slope mean: %.3f\n" % np.mean(all_slopes))

            pos = [(label, slope) for label, slope, _, corr, pval in lines if pval < 0.05 and corr > 0]
            neg = [(label, slope) for label, slope, _, corr, pval in lines if pval < 0.05 and corr < 0]

            if pos:
                mean_pos = np.mean([s for _, s in pos])
                f.write("positive correlation (p < 0.05): mean = %.3f\n" % mean_pos)
                for label, slope in pos:
                    f.write("  - %s: %.3f\n" % (label, slope))

            if neg:
                f.write("\n")
                mean_neg = np.mean([s for _, s in neg])
                f.write("negative correlation (p < 0.05): mean = %.3f\n" % mean_neg)
                for label, slope in neg:
                    f.write("  - %s: %.3f\n" % (label, slope))

    write_corr(CORR_ACC_FILE, acc_results)
    write_corr(CORR_FAIR_FILE, fair_results)
    write_regression(REGRESSION_ACC_FILE, regression_acc_lines)
    write_regression(REGRESSION_FAIR_FILE, regression_fair_lines)

    print("Saved: %s, %s" % (CORR_ACC_FILE, CORR_FAIR_FILE))
    print("Saved: %s, %s" % (REGRESSION_ACC_FILE, REGRESSION_FAIR_FILE))

def main():
    run_correlation()

if __name__ == "__main__":
    main()
