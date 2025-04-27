from __future__ import division
import os
import sys
import numpy as np
import random
import time
import datetime
import pandas as pd
from scipy.optimize import basinhopping
from config import config_census
from config import config_german
from config import config_bank
import copy
import heapq
from CT import generateCTFiles
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


###  before retrain  ###

random.seed(time.time())
dataset, sensitive_name = sys.argv[1].split("_")
classifier = sys.argv[2]

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

"""   
if classifier == "SVM":
    model = SVC(gamma=0.0025)
elif classifier == "MLPC":
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                        alpha=0.0001, batch_size='auto', learning_rate='constant',
                        learning_rate_init=0.001, power_t=0.5, max_iter=200,
                        shuffle=True, random_state=42, tol=0.0001,
                        verbose=False, warm_start=False, momentum=0.9,
                        nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                        beta_1=0.9, beta_2=0.999, epsilon=1e-08)  
elif classifier == "RF":
    model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                max_depth=5, max_features='auto', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                                oob_score=False, random_state=42, verbose=0,
                                warm_start=False)
else:
    print "The classifier is wrong."

model.fit(train_X,train_Y)
"""

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

"""
SVM = SVC(gamma=0.0025)

MLPC = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                        alpha=0.0001, batch_size='auto', learning_rate='constant',
                        learning_rate_init=0.001, power_t=0.5, max_iter=200,
                        shuffle=True, random_state=42, tol=0.0001,
                        verbose=False, warm_start=False, momentum=0.9,
                        nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                        beta_1=0.9, beta_2=0.999, epsilon=1e-08)
RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                max_depth=5, max_features='auto', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                                oob_score=False, random_state=42, verbose=0,
                                warm_start=False)
SVM.fit(train_X,train_Y)
MLPC.fit(train_X,train_Y)
RF.fit(train_X,train_Y)
"""

def get_pair_list(discs):
        pair_list = []
        for disc in discs:
            disc_0 = [int(i) for i in disc]
            disc_1 = [int(i) for i in disc]
            disc_0[sensitive_param - 1] = input_bounds[sensitive_param - 1][0]
            disc_1[sensitive_param - 1] = input_bounds[sensitive_param - 1][1]
            pair_list.append(disc_0)
            pair_list.append(disc_1)
        return pair_list


def majority_voting(inp, models):
    inp = [int(i) for i in inp]
    inp = np.asarray(inp)
    inp = np.reshape(inp, (1, -1))
    """
    voting = [SVM.predict(inp)[0], MLPC.predict(inp)[0], RF.predict(inp)[0]]
    """
    voting = [int(m.predict(inp)[0]) for m in models]
    counts = np.bincount(voting)
    label = np.argmax(counts)
    return label


majority_models = [get_model("SVM"), get_model("MLPC"), get_model("RF")]
for m in majority_models:
    m.fit(train_X, train_Y)


def L1_distance(disc1,disc2):
    distance = 0
    for i in range(len(disc1)):
        difference = abs(disc1[i]-disc2[i])
        if dataset=="CENSUS":
            distance += float(difference)/(config_census.input_bounds[i][1]- config_census.input_bounds[i][0])
        elif dataset=="GERMAN":
            distance += float(difference)/(config_german.input_bounds[i][1]- config_german.input_bounds[i][0])
        elif dataset=="BANK":
            distance += float(difference)/(config_bank.input_bounds[i][1]- config_bank.input_bounds[i][0])
        else:
            print("wrong dataset")
            sys.exit(1)
    return distance


def get_discs(DISC,diversity,disc_num):
    DISC_copy = copy.deepcopy(DISC)
    discs=[]
    random.seed(time.time())
    added = random.choice(DISC_copy)
    discs.append(added)
    DISC_copy.remove(added)
    
    for i in range(disc_num-1):
        random.seed(time.time())
        candidates = random.sample(DISC_copy,100)
        min_distances = []
        
        for candidate in candidates:   
            distances = []
            for disc in discs:               
                distances.append(L1_distance(candidate,disc))
            min_distances.append(min(distances))
        
        added_distance = heapq.nsmallest(diversity,min_distances)[-1]
        
        added_index = min_distances.index(added_distance)
        
        added = candidates[added_index]
        discs.append(added)
        DISC_copy.remove(added)
    
    return discs


random.seed(time.time())
added_num = int(train_data_size * 0.05)

for diversity in [1,10,20,30,40,50,60,70,80,90,100]:
    print("==== Diversity Information ====")
    print("Diversity Level: {} (Max: 100)".format(diversity))
    print("===============================")
    """
    if diversity==0:
        added_disc = random.choice(disc_inputs_list) 
        added_disc_list = []
        for i in range(added_num):
            added_disc_list.append(added_disc)
    elif diversity=="random":
        added_disc_list = random.sample(disc_inputs_list, added_num)
    else:
        added_disc_list = get_discs(disc_inputs_list,diversity,added_num) 
    """
    added_disc_list = get_discs(disc_inputs_list,diversity,added_num) 
    
    
    # check distance
    
    if len(added_disc_list)<=1000:
        disc_num = len(added_disc_list)     
    else:
        disc_num = 1000
    
    disc = added_disc_list[0:disc_num]
    distance = 0
    count=0
    for i in range(0,disc_num-1):
        for j in range(i+1,disc_num):
            distance += L1_distance(disc[i],disc[j])
            count += 1
    distance = float(distance) / count

    added_disc = np.array(get_pair_list(added_disc_list))
    
    added_X = added_disc
    
    added_Y = []
    for disc in added_disc_list:
        label = majority_voting(disc, majority_models)
        added_Y.append(label)
        added_Y.append(label)
    added_Y = np.array(added_Y)
    
    new_train_X = np.concatenate((train_X, added_X), axis = 0)  
    new_train_Y = np.concatenate((train_Y, added_Y), axis = 0)

    """
    if classifier == "SVM":
        retrained_model = SVC(gamma=0.0025)
    elif classifier == "MLPC":
        retrained_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                            alpha=0.0001, batch_size='auto', learning_rate='constant',
                            learning_rate_init=0.001, power_t=0.5, max_iter=200,
                            shuffle=True, random_state=42, tol=0.0001,
                            verbose=False, warm_start=False, momentum=0.9,
                            nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)   
    elif classifier == "RF":
        retrained_model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                    max_depth=5, max_features='auto', max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=1, min_samples_split=2,
                                    min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                                    oob_score=False, random_state=42, verbose=0,
                                    warm_start=False)
    else:
        print "The classifier is wrong."
        
    """

    retrained_model = get_model(classifier)
    retrained_model.fit(new_train_X,new_train_Y)  
    after_accuracy = accuracy(retrained_model,test_data)
    after_fairness = np.mean(get_estimate_array(retrained_model))
    
    print("After retrain - Accuracy: {:.6f}".format(after_accuracy))
    print("After retrain - Fairness (IFr): {:.6f}".format(after_fairness))

    save_path = "diversity_retrain_results/{}_{}/{}/{}_{}_{}_{}_{}.txt".format(
        dataset, sensitive_name, classifier, dataset, sensitive_name, classifier, str(N), str(diversity)
    )
    with open(save_path, "a") as myfile:
        myfile.write(str(len(disc_inputs_list)) + " "
                    + str(before_accuracy) + " "
                    + str(before_fairness) + " "
                    + str(after_accuracy) + " "
                    + str(after_fairness) + " "
                    + str(distance) + " "
                    + "\n"
                    )

    print("Results saved to: {}\n".format(save_path))
