import subprocess

dataset_protected_pairs = [
    ("Adult", "age"),
    ("Adult", "race"),
    ("Adult", "sex"),
    ("Bank", "age"),
    ("Credit", "age"),
    ("Credit", "sex"),
]

models = ["SVM", "MLP", "RanForest"]

for dataset, protected_attr in dataset_protected_pairs:
    for model in models:
        cmd = [
            "python", "exp.py",
            "--method", "aft",
            "--dataset_name", dataset,
            "--protected_attr", protected_attr,
            "--model_name", model,
            "--repeat", "30",
            "--runtime", "3600"
        ]
        print(f"Running: {dataset}-{protected_attr}-{model}")
        subprocess.run(cmd)
