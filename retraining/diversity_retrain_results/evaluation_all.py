import evaluation
import os

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".txt"):
            filename = os.path.splitext(file)[0]  # "BANK_age_SVM_10000_1"
            parts = filename.split("_")  # ['BANK', 'age', 'SVM', '10000', '1']

            if len(parts) != 5:
                continue

            dataset = parts[0].upper()          # BANK
            sensitive_param = parts[1]          # age
            model = parts[2]                    # SVM
            N = parts[3]                        # 10000
            diversity = parts[4]                # 1

            evaluation.evaluation(diversity, dataset, sensitive_param, model, N)

