import evaluation
import os

for algorithm in ["const_const", "const_random", "const_majority", "random_const", "random_random", "random_majority", "pair_const", "pair_random", "pair_majority"]:
    for dataset in ["CENSUS_sex", "CENSUS_race", "CENSUS_age", "GERMAN_sex", "GERMAN_age", "BANK_age"]:
        for classifier in ["SVM", "MLPC", "RF"]:
            for N in ["100", "200", "300", "500", "1000", "2000", "5000", "10000", "20000", "40000", "100000"]:
                path = algorithm + "/" + dataset + "_" + classifier + "_" + N + ".txt"
                if os.path.isfile(path):
                    evaluation.evaluation(algorithm, dataset, classifier, N)