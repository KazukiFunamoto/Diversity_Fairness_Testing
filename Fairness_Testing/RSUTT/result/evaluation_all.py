import evaluation
import os

for algorithm in ["AEQUITAS", "CGFT", "KOSEI", "RSUTT"]:
    for dataset in ["CENSUS", "GERMAN", "BANK"]:
        for classifier in ["DT", "MLPC", "RF"]:
            for N in ["3000", "7000"]:
                path = algorithm + "/" + dataset + "_" + classifier + "_" + N + ".txt"
                if os.path.isfile(path):
                    evaluation.evaluation(algorithm, dataset, classifier, N)
