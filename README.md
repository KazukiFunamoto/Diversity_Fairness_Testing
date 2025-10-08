# Diversity_Fairness_Testing

## Publication

This repository supports the paper:

**_"Is Diversity a Meaningful Metric in Fairness Testing?"_** 
(*Accepted to and Presented at the [ACM/IEEE International Symposium on Empirical Software Engineering and Measurement (ESEM 2025)](https://conf.researchr.org/home/esem-2025), Technical Track*)

Kazuki Funamoto (船本 和希) – Keio University  
Takashi Kitamura (北村 崇師) – National Institute of Advanced Industrial Science and Technology (AIST)  
Shingo Takada (高田 眞吾) – Keio University  

## About this repository

This repository contains the code, datasets, and experimental results for the paper *"Is Diversity a Meaningful Metric in Fairness Testing?"*.  
The repository is organized as follows:

- `all_results/`: Contains the results of all experiments, covering RQ, RQ1, RQ2, and RQ3.
- `fairness_testings/`: Contains scripts and folders for executing fairness testing algorithms (RSUTT, AFT, THEMIS), detecting discriminatory data, and measuring their diversity.
- `retraining/`: Contains scripts and folders for retraining experiments to investigate how IDI diversity affects fairness improvement and accuracy degradation, as well as to compare different retraining methods.

The experiments were conducted using Python 2.7.18.  
However, AFT and THEMIS require Python 3.8.10, with different dependencies, which are explained in `fairness_testings/README.md`.

The general dependencies are listed in `requirements.txt`.  
You can install them by running:

```
pip install -r requirements.txt
```

The file `add_gitkeep_to_empty_dirs.py` was created to allow empty directories (such as `fairness_testings/distance_results/`, `retraining/diversity_retrain_results/`, and `retraining/retrain_methods_results/`) to be committed to Git.  
This file is only for that purpose and can be ignored.

Each major folder (`all_results/`, `fairness_testings/`, and `retraining/`) contains its own `README.md` file.
Please refer to the corresponding `README.md` in each folder for more information.

## License

This code is licensed under the MIT License.  

