import subprocess

execution_times = 30

dataset_protected_pairs = [
    ("BANK", "age"),
    ("CENSUS", "age"),
    ("CENSUS", "race"),
    ("CENSUS", "sex"),
    ("GERMAN", "age"),
    ("GERMAN", "sex"),
]

models = ["SVM", "MLPC", "RF"]

for i in range(execution_times):
    for dataset, attr in dataset_protected_pairs:
        for model in models:
            cmd = "python retrain_methods.py all {}_{} {}".format(dataset, attr, model)
            subprocess.call(cmd, shell=True)

