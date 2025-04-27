from __future__ import division
import os
import sys
import numpy as np
import random
import time
import datetime
import pandas as pd
from scipy.optimize import basinhopping
import copy
from CT import generateCTFiles
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

###  before retrain  ###

SCENARIOS = [
    "const_const", "const_random", "const_majority",
    "random_const", "random_random", "random_majority",
    "pair_const", "pair_random", "pair_majority"
]

mode = sys.argv[1]  # e.g. "all" or one of SCENARIOS

random.seed(time.time())
dataset, sensitive_name = sys.argv[2].split("_")
classifier = sys.argv[3]

def decide_N(dataset, protected_attr, model):
    if dataset == "BANK":
        return 1000 if model == "RF" else 10000
    if dataset == "CENSUS":
        if protected_attr == "age":
            return 1000
        elif protected_attr == "race":
            return 1000 if model == "RF" else 10000
        elif protected_attr == "sex":
            return 100000 if model == "SVM" else (10000 if model == "MLPC" else 1000)
    if dataset == "GERMAN":
        if protected_attr == "age":
            return 1000
        elif protected_attr == "sex":
            return 10000 if model == "SVM" else 1000
    raise ValueError("Invalid dataset or protected attribute")

N = decide_N(dataset, sensitive_name, classifier)

if dataset == "CENSUS":
    from config import config_census as config
elif dataset == "GERMAN":
    from config import config_german as config
    if sensitive_name not in ["age", "sex"]:
        raise ValueError("Invalid sensitive name for GERMAN")
elif dataset == "BANK":
    from config import config_bank as config
    if sensitive_name != "age":
        raise ValueError("Invalid sensitive name for BANK")
else:
    raise ValueError("Invalid dataset name")

input_bounds = config.input_bounds
threshold = config.threshold
perturbation_unit = config.perturbation_unit
params = config.params
if sensitive_name == "age":
    sensitive_param = config.sensitive_param_age
elif sensitive_name == "race":
    sensitive_param = config.sensitive_param_race
elif sensitive_name == "sex":
    sensitive_param = config.sensitive_param_sex
else:
    raise ValueError("Invalid sensitive name")

print("==== Experiment Configuration ====")
print("Dataset: {}".format(dataset))
print("Protected Attribute: {}".format(sensitive_name))
print("Classifier: {}".format(classifier))
print("==================================")

path = "classifier/datasets/{}.csv".format(dataset.lower())
df = pd.read_csv(path) 
data = df.values
np.random.shuffle(data)

data_size = len(data)
train_data_size = int(data_size * 0.8)

train_data = data[0:train_data_size]
test_data = data[train_data_size:]

train_X = train_data[:, :-1]
train_Y = train_data[:, -1]
test_X = test_data[:, :-1]
test_Y = test_data[:, -1]


def get_model(model_type):   
    if model_type == "SVM":
        return SVC(gamma=0.0025)
    elif model_type == "MLPC":
        return MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                            alpha=0.0001, batch_size='auto', learning_rate='constant',
                            learning_rate_init=0.001, power_t=0.5, max_iter=200,
                            shuffle=True, random_state=42, tol=0.0001,
                            verbose=False, warm_start=False, momentum=0.9,
                            nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)   
    elif model_type == "RF":
        return RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                    max_depth=5, max_features='auto', max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=1, min_samples_split=2,
                                    min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                                    oob_score=False, random_state=42, verbose=0,
                                    warm_start=False)
    else:
        raise ValueError("Invalid classifier")

model = get_model(classifier)
model.fit(train_X,train_Y)

def accuracy(model,test_data):
    num = len(test_data)
    correct_num = 0
    for data in test_data:
        test_x = data[:-1]
        test_y = data[-1]
        test_x = np.reshape(test_x, (1, -1))
        if model.predict(test_x) == test_y:
            correct_num += 1
    accuracy = float(correct_num)/num
    return accuracy
        
before_accuracy = accuracy(model,test_data)

num_trials = 400
samples = 100

def get_random_input(): 
    x = []
    for i in xrange(params):
        x.append(random.randint(input_bounds[i][0], input_bounds[i][1]))
    x[sensitive_param - 1] = 0
    return x

def evaluate_input(inp,model):
    inp0 = [int(i) for i in inp]
    inp1 = [int(i) for i in inp]

    inp0[sensitive_param - 1] = input_bounds[sensitive_param - 1][0]
    inp1[sensitive_param - 1] = input_bounds[sensitive_param - 1][1]

    inp0 = np.asarray(inp0)
    inp0 = np.reshape(inp0, (1, -1))

    inp1 = np.asarray(inp1)
    inp1 = np.reshape(inp1, (1, -1))

    out0 = model.predict(inp0)
    out1 = model.predict(inp1)
    
    return (out0!=out1)

def get_estimate_array(model):
    random.seed(time.time())    
    estimate_array = []
    rolling_average = 0.0
    for i in xrange(num_trials):
        disc_count = 0
        total_count = 0
        for j in xrange(samples):
            total_count = total_count + 1
            if(evaluate_input(get_random_input(),model)):
                disc_count = disc_count + 1

        estimate = float(disc_count)/total_count
        rolling_average = ((rolling_average * i) + estimate)/(i + 1)
        estimate_array.append(estimate) 
    return estimate_array
 
before_fairness = np.mean(get_estimate_array(model))

print("Before retrain - Accuracy: {:.6f}".format(before_accuracy))
print("Before retrain - Fairness (IFr): {:.6f}".format(before_fairness))


###  RSUTT algorithm
disc_inputs = set()
disc_inputs_list = []

tot_inputs = set()

global_iteration_limit = N
local_iteration_limit = 0
local_cnt = 0


class GlobalDiscovery(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, x):
        for i in xrange(params):
            random.seed(time.time())
            x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])

        x[sensitive_param - 1] = input_bounds[sensitive_param - 1][0]
        return x


def evaluate_disc(inp):
    inp0 = [int(i) for i in inp]
    inp1 = [int(i) for i in inp]

    inp0[sensitive_param - 1] = input_bounds[sensitive_param - 1][0]
    inp1[sensitive_param - 1] = input_bounds[sensitive_param - 1][1]

    inp0 = np.asarray(inp0)
    inp0 = np.reshape(inp0, (1, -1))

    inp1 = np.asarray(inp1)
    inp1 = np.reshape(inp1, (1, -1))

    out0 = model.predict(inp0)
    out1 = model.predict(inp1)
    tot_inputs.add(tuple(map(tuple, inp0)))

    if abs(out0 - out1) > threshold and tuple(map(tuple, inp0)) not in disc_inputs:
        disc_inputs.add(tuple(map(tuple, inp0)))
        disc_inputs_list.append(inp0.tolist()[0])

    # return not abs(out0 - out1) > threshold
    # for binary classification, we have found that the
    # following optimization function gives better results
    return abs(out1 + out0)

# Local search algorithm for each dataset

def my_local_search(inp):
    for param in range(params):
        if param == sensitive_param - 1:
            continue
        else:
            for direction in [-1, 1]:
                inp2 = copy.copy(inp)
                inp2[param] = inp2[param] + direction
                if inp2[param] < input_bounds[param][0] and direction == -1:
                    continue
                elif inp2[param] > input_bounds[param][1] and direction == 1:
                    continue
                elif tuple(map(tuple, np.reshape(np.asarray(inp2), (1, -1)))) in tot_inputs:
                    continue
                evaluate_disc(inp2)
                global local_cnt
                local_cnt += 1

starting_time = time.time()
minimizer = {"method": "L-BFGS-B"}

global_discovery = GlobalDiscovery()

def extract_testcases(filename):
    x = []
    i = 0
    with open(filename, "r") as ins:
        for line in ins:
            if i < 7:
                i = i + 1
                continue
            line = line.strip()
            line1 = line.split(',')
            y = map(int, line1)
            x.append(y)
    return x


def obtain_ct_length(t):
        # Returns TS in good format and length
        # If this test suite does not exits, create
        strength = str(t)
        if os.path.exists("CT/" + dataset.lower() + "/" + dataset.lower() + "TS" + strength + "w.csv"):
            pass
        else:
            generateCTFiles.generateCT(dataset, t)

        tsaux = extract_testcases("CT/" + dataset.lower() + "/" + dataset.lower() + "TS" + strength + "w.csv")
        tslen = len(tsaux)

        return tsaux, tslen


def select_from_ctfile(number_of_inputs):

    global test_suite_base

    # Initialize the test suite base
    test_suite_base, length_base = obtain_ct_length(1)
    test_suite_alpha, length_alpha = obtain_ct_length(2)

    i = 3

    # Case 1: fewer inputs than length base
    if number_of_inputs <= length_base:
        # Obtain all test cases from
        test_suite_CT_Extra, _, _, _ = train_test_split(test_suite_base, [0] * len(test_suite_base),
                                                        test_size=len(test_suite_base) - int(number_of_inputs))
        return test_suite_CT_Extra

    # Case 2: combine two test suites
    # Define base and alpha TS
    while length_alpha < number_of_inputs:
        test_suite_base = test_suite_alpha
        length_base = length_alpha
        test_suite_alpha, length_alpha = obtain_ct_length(i)
        i = i + 1

    set_base = set(tuple(a) for a in test_suite_base)
    set_alpha = set(tuple(a) for a in test_suite_alpha)

    setDifference = set_alpha - set_base
    listDifference = list(setDifference)
    difference_array = np.array(listDifference)

    # Number of inputs to be added
    n_alpha = number_of_inputs - len(test_suite_base)

    # Inputs to be added
    test_suite_CT_Extra, _, _, _ = train_test_split(difference_array, [0] * len(difference_array),
                                                    test_size=len(difference_array) - int(n_alpha))

    # Add them
    test_suite_CT_Selected = test_suite_base
    test_suite_CT_Selected = np.append(test_suite_CT_Selected, test_suite_CT_Extra, axis=0)

    return test_suite_CT_Selected

test_suite_CT_Selected = select_from_ctfile(global_iteration_limit)

for inp in test_suite_CT_Selected:
    evaluate_disc(inp)

seedData = len(disc_inputs_list)

local_iteration_limit = seedData * 2000

for input in disc_inputs_list:
    if local_cnt < local_iteration_limit:
        my_local_search(input)
    else:
        break

#print "Total evaluated data: " + str(len(tot_inputs))
#print "Number of seed data: " + str(seedData)
#print "Number of discriminatory data: " + str(len(disc_inputs_list))
#print "Percentage of discriminatory data: " + str(float(len(disc_inputs_list)) / float(len(tot_inputs)) * 100)



###   retrain   ###

def get_pair_list(discs):
    pair_list = []
    for disc in discs:
        d0, d1 = list(disc), list(disc)        
        d0[sensitive_param - 1] = input_bounds[sensitive_param - 1][0]
        d1[sensitive_param - 1] = input_bounds[sensitive_param - 1][1]
        pair_list.append(d0)
        pair_list.append(d1)
    return np.array(pair_list)


def majority_voting(inp, models):
    inp = list(inp)
    inp = np.array(inp).reshape(1, -1)
    
    votes = [int(m.predict(inp)[0]) for m in models]
    counts = np.bincount(votes)
    label = np.argmax(counts)
    
    return label

majority_models = [get_model("SVM"), get_model("MLPC"), get_model("RF")]
for m in majority_models:
    m.fit(train_X, train_Y)

def run_scenario(mode):
    
    print("==== Retraining Methods Information ====")
    print("Selected Method: {}".format(mode))
    print("===============================")
    
    base_ratio = 0.05 if mode.startswith("pair") else 0.1
    added_num = int(train_data_size * base_ratio)
    discs = random.sample(disc_inputs_list, added_num)
    
    # X

    if mode.startswith("pair"):
        X = get_pair_list(discs)
    else:
        X = np.array([list(d) for d in discs])
        if mode.startswith("random"):
            for x in X:
                x[sensitive_param - 1] = random.choice([input_bounds[sensitive_param - 1][0], input_bounds[sensitive_param - 1][1]])
        elif mode.startswith("const"):  
            for x in X:
                x[sensitive_param - 1] = input_bounds[sensitive_param - 1][0]
                
    
    # Y
    
    label_values = np.unique(train_Y)
    
    if mode.endswith("majority"):
        Y = []
        if mode.startswith("pair"):
            for i in range(0, len(X), 2):
                label = majority_voting(X[i], majority_models)
                Y.extend([label, label])
        else:
            for x in X:
                label = majority_voting(x, majority_models)
                Y.append(label)
    elif mode.endswith("random"):
        if mode.startswith("pair"):
            Y = []
            for _ in range(len(X)//2):  # (X is even)
                label = random.choice(label_values)
                Y.extend([label, label])
        else:
            Y = [random.choice(label_values) for _ in range(len(X))]    
    elif mode.endswith("const"):
        Y = [label_values[0] for _ in range(len(X))]

    new_X = np.concatenate((train_X, X), axis=0)
    new_Y = np.concatenate((train_Y, Y), axis=0)

    retrained_model = get_model(classifier)
    retrained_model.fit(new_X, new_Y)
    after_accuracy = accuracy(retrained_model, test_data)
    after_fairness = np.mean(get_estimate_array(retrained_model))
    
    print("After retrain - Accuracy: {:.6f}".format(after_accuracy))
    print("After retrain - Fairness (IFr): {:.6f}".format(after_fairness))

    outdir = "retrain_methods_results/{}".format(mode)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = "{}/{}_{}_{}_{}.txt".format(outdir, dataset, sensitive_name, classifier, N)
    with open(outpath, "a") as f:
        f.write("{} {} {} {} {}\n".format(len(disc_inputs_list), before_accuracy, before_fairness, after_accuracy, after_fairness))
    print("Results saved to: {}\n".format(outpath))

if mode == "all":
    for m in SCENARIOS:
        run_scenario(m)
else:
    if mode not in SCENARIOS:
        raise ValueError("Invalid mode")
    run_scenario(mode)

