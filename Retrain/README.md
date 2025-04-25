# RSUTT

# Execution Environment

programming language

* Python : 2.7.18

expansion module

* numpy : 1.14.5
* pandas : 0.24.2
* scipy : 1.1.0
* scikit-learn : 0.19.0

The authors used Pycharm CE 2022.2.3 as the development IDE.

# Scikit-Learn Classifiers

* Decision Tree(DT)
* MLPC
* Random Forest(RF)

Each of the following three directories contains the following

It is in the "classifier" directory.

* Adult (Census Income) : Scikit-Learn classifiers in the "census" directory
* Statlog (German Credit)：Scikit-Learn classifiers in the "german" directory
* Bank Marketing : Scikit-Learn classifiers in the "bank" directory

# Config 

It is in the "config" directory.

* Adult (Census Income) : config_census.py
* Statlog(German Credit)：config_german.py
* Bank Marketing : config_bank.py

The config files has the following data (same as the Udeshi's codes) : 

* params : The number of parameters in the data
* sensitive_param : The parameter under test
* input_bounds : The bounds of each parameter
* classifier_name : Pickled scikit-learn classifier under test (only applicable to the sklearn files)
* threshold : Discrimination threshold
* perturbation_unit : By what unit would the user like to perturb the input in the local search
* retraining_inputs : Inputs to be used for the retraining

# Demo

`python RSUTT.py <algorithm> <classifier> <dataset> <N>`

first parameter : algorithm

* AEQUITAS 
* KOSEI 
* CGFT 
* RSUTT

second parameter : classifier

* DT 
* MLPC 
* RF

third parameter : dataset

* CENSUS
* GERMAN
* BANK

fourth parameter : N

N is global_iteration_limit.


# iteration
We set the default iteration limits in the following:
global_iteration_limit = N
local_iteration_limit = 2000
