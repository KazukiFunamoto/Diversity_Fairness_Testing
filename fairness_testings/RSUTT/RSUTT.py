from __future__ import division
import os
import sys
import numpy as np
import random
import time
import datetime
from scipy.optimize import basinhopping
from config import config_census
from config import config_german
from config import config_bank
import copy
from CT import generateCTFiles
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
sys.path.insert(0, './fair_classification/')    

random.seed(time.time())
algorithm = sys.argv[1]
dataset, sensitive_name = sys.argv[2].split("_")
classifier = sys.argv[3]
repeat_times = int(sys.argv[4])

print "start time : " + str(datetime.datetime.now())

# Setting for each Dataset

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
    params = config_census.params
    if sensitive_name == "age":
        sensitive_param = config_census.sensitive_param_age
    elif sensitive_name == "race":
        sensitive_param = config_census.sensitive_param_race
    elif sensitive_name == "sex":
        sensitive_param = config_census.sensitive_param_sex
    else:
        print "error"
        sys.exit(1)
    perturbation_unit = config_census.perturbation_unit
    threshold = config_census.threshold
    input_bounds = config_census.input_bounds
    dataset_num = 32561
    if classifier == "SVM":
        classifier_name = 'classifier/census/SVM_CENSUS.pkl'
    elif classifier == "MLPC":
        classifier_name = 'classifier/census/MLPC_CENSUS.pkl'
    elif classifier == "RF":
        classifier_name = 'classifier/census/RF_CENSUS.pkl'
    else:
        print "The classifier is wrong."

elif dataset == "GERMAN":
    params = config_german.params
    if sensitive_name == "sex":
        sensitive_param = config_german.sensitive_param_sex
    elif sensitive_name == "age":
        sensitive_param = config_german.sensitive_param_age
    else:
        print "error"
        sys.exit(1)
    perturbation_unit = config_german.perturbation_unit
    threshold = config_german.threshold
    input_bounds = config_german.input_bounds
    dataset_num = 1000
    if classifier == "SVM":
        classifier_name = 'classifier/german/SVM_GERMAN.pkl'
    elif classifier == "MLPC":
        classifier_name = 'classifier/german/MLPC_GERMAN.pkl'
    elif classifier == "RF":
        classifier_name = 'classifier/german/RF_GERMAN.pkl'
    else:
        print "The classifier is wrong."

elif dataset == "BANK":
    params = config_bank.params
    if sensitive_name == "age":
        sensitive_param = config_bank.sensitive_param_age
    else:
        print "error"
        sys.exit(1)
    perturbation_unit = config_bank.perturbation_unit
    threshold = config_bank.threshold
    input_bounds = config_bank.input_bounds
    dataset_num = 45211
    if classifier == "SVM":
        classifier_name = 'classifier/bank/SVM_BANK.pkl'
    elif classifier == "MLPC":
        classifier_name = 'classifier/bank/MLPC_BANK.pkl'
    elif classifier == "RF":
        classifier_name = 'classifier/bank/RF_BANK.pkl'
    else:
        print "The classifier is wrong."

else:
    print "The dataset name is wrong."

# Aequitas algorithm

if algorithm == "AEQUITAS":
    init_prob = 0.5
    direction_probability = [init_prob] * params
    direction_probability_change_size = 0.001

    param_probability = [1.0 / params] * params
    param_probability_change_size = 0.001

    global_disc_inputs = set()
    global_disc_inputs_list = []

    local_disc_inputs = set()
    local_disc_inputs_list = []

    tot_inputs = set()

    global_iteration_limit = N
    local_iteration_limit = 2000
    model = joblib.load(classifier_name)


    def normalise_probability():
        probability_sum = 0.0
        for prob in param_probability:
            probability_sum = probability_sum + prob

        for i in range(params):
            param_probability[i] = float(param_probability[i]) / float(probability_sum)


    class LocalPerturbation(object):

        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            param_choice = np.random.choice(xrange(params), p=param_probability)
            act = [-1, 1]
            direction_choice = np.random.choice(act, p=[direction_probability[param_choice],
                                                        (1 - direction_probability[param_choice])])

            if (x[param_choice] == input_bounds[param_choice][0]) or (x[param_choice] == input_bounds[param_choice][1]):
                direction_choice = np.random.choice(act)

            x[param_choice] = x[param_choice] + (direction_choice * perturbation_unit)

            x[param_choice] = max(input_bounds[param_choice][0], x[param_choice])
            x[param_choice] = min(input_bounds[param_choice][1], x[param_choice])

            ei = evaluate_input(x)

            if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
                direction_probability[param_choice] = min(
                    direction_probability[param_choice] + (direction_probability_change_size * perturbation_unit), 1)

            elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
                direction_probability[param_choice] = max(
                    direction_probability[param_choice] - (direction_probability_change_size * perturbation_unit), 0)

            if ei:
                param_probability[param_choice] = param_probability[param_choice] + param_probability_change_size
                normalise_probability()
            else:
                param_probability[param_choice] = max(param_probability[param_choice] - param_probability_change_size,
                                                      0)
                normalise_probability()

            return x


    class GlobalDiscovery(object):
        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            for i in xrange(params):
                random.seed(time.time())
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])

            x[sensitive_param - 1] = input_bounds[sensitive_param - 1][0]
            return x


    def evaluate_input(inp):
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

        # return (abs(out0 - out1) > threshold)
        # for binary classification, we have found that the
        # following optimization function gives better results
        return abs(out1 + out0)


    def evaluate_global(inp):
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

        if abs(out0 - out1) > threshold and tuple(map(tuple, inp0)) not in global_disc_inputs:
            global_disc_inputs.add(tuple(map(tuple, inp0)))
            global_disc_inputs_list.append(inp0.tolist()[0])

        # return not abs(out0 - out1) > threshold
        # for binary classification, we have found that the
        # following optimization function gives better results
        return abs(out1 + out0)


    def evaluate_local(inp):
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

        if (abs(out0 - out1) > threshold and (tuple(map(tuple, inp0)) not in global_disc_inputs)
                and (tuple(map(tuple, inp0)) not in local_disc_inputs)):
            local_disc_inputs.add(tuple(map(tuple, inp0)))
            local_disc_inputs_list.append(inp0.tolist()[0])

        # return not abs(out0 - out1) > threshold
        # for binary classification, we have found that the
        # following optimization function gives better results
        return abs(out1 + out0)

    print "Search started"
    starting_time = time.time()
    if dataset == "CENSUS":
        initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
    elif dataset == "GERMAN":
        initial_input = [1, 4, 1, 23, 4, 4, 0, 1, 2, 20, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1]
    elif dataset == "BANK":
        initial_input = [0, 5, 1, 2, 0, 23, 0, 1, 2, 20, 5, 35, 39, 1, 1, 2]
    minimizer = {"method": "L-BFGS-B"}

    global_discovery = GlobalDiscovery()
    local_perturbation = LocalPerturbation()

    basinhopping(evaluate_global, initial_input, stepsize=1.0, take_step=global_discovery, minimizer_kwargs=minimizer,
                 niter=global_iteration_limit)

    seedData = len(global_disc_inputs_list)

    for input in global_disc_inputs_list:
        basinhopping(evaluate_local, input, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer,
                     niter=local_iteration_limit)

    print "Total evaluated data: " + str(len(tot_inputs))
    print "Number of seed data: " + str(seedData)
    print "Number of discriminatory data: " + str(len(global_disc_inputs_list) + len(local_disc_inputs_list))
    print "Percentage of discriminatory data: " + str(float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) 
                                                      / float(len(tot_inputs)) * 100)
    elapsed_time = time.time() - starting_time
    print "Number of discriminatory data per second: " \
          + str((len(global_disc_inputs_list) + len(local_disc_inputs_list)) / elapsed_time)
    print ("Execution_time: {0}".format(elapsed_time) + "[sec]")

    with open("result/" + algorithm + "/" + dataset + "_" + classifier + "_" + str(N) + ".txt", "a") as myfile:
        myfile.write(str(len(tot_inputs)) + " "
                     + str(seedData) + " "
                     + str(len(global_disc_inputs_list) + len(local_disc_inputs_list)) + " "
                     + str(float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                           / float(len(tot_inputs)) * 100) + " "
                     + str((len(global_disc_inputs_list) + len(local_disc_inputs_list)) / elapsed_time) + " "
                     + "{0}".format(elapsed_time) + " "
                     + "\n"
                     )

    print "Search ended"

# KOSEI algorithm

elif algorithm == "KOSEI":
    disc_inputs = set()
    disc_inputs_list = []

    tot_inputs = set()

    global_iteration_limit = N
    local_iteration_limit = 0
    local_cnt = 0

    model = joblib.load(classifier_name)


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

    def my_local_search_census(inp):
        for param in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]:
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


    def my_local_search_german(inp):
        for param in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
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


    def my_local_search_bank(inp):
        for param in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
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


    print "Search started"
    starting_time = time.time()
    if dataset == "CENSUS":
        initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
    elif dataset == "GERMAN":
        initial_input = [1, 4, 1, 23, 4, 4, 0, 1, 2, 20, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1]
    elif dataset == "BANK":
        initial_input = [0, 5, 1, 2, 0, 23, 0, 1, 2, 20, 5, 35, 39, 1, 1, 2]
    minimizer = {"method": "L-BFGS-B"}

    global_discovery = GlobalDiscovery()

    basinhopping(evaluate_disc, initial_input, stepsize=1.0, take_step=global_discovery, minimizer_kwargs=minimizer,
                 niter=global_iteration_limit)

    seedData = len(disc_inputs_list)

    local_iteration_limit = seedData * 2000

    for input in disc_inputs_list:
        if local_cnt < local_iteration_limit:
            if dataset == "CENSUS":
                my_local_search_census(input)
            elif dataset == "GERMAN":
                my_local_search_german(input)
            elif dataset == "BANK":
                my_local_search_bank(input)
        else:
            break

    print "Total evaluated data: " + str(len(tot_inputs))
    print "Number of seed data: " + str(seedData)
    print "Number of discriminatory data: " + str(len(disc_inputs_list))
    print "Percentage of discriminatory data: " + str(float(len(disc_inputs_list)) / float(len(tot_inputs)) * 100)
    elapsed_time = time.time() - starting_time
    print "Number of discriminatory data per second: " + str(len(disc_inputs_list) / elapsed_time)
    print ("Execution_time:{0}".format(elapsed_time) + "[sec]")

    with open("result/" + algorithm + "/" + dataset + "_" + classifier + "_" + str(N) + ".txt", "a") as myfile:
        myfile.write(str(len(tot_inputs)) + " "
                     + str(seedData) + " "
                     + str(len(disc_inputs_list)) + " "
                     + str(float(len(disc_inputs_list))
                           / float(len(tot_inputs)) * 100) + " "
                     + str(len(disc_inputs_list) / elapsed_time) + " "
                     + "{0}".format(elapsed_time) + " "
                     + "\n"
                     )

    print "Search ended"

# CGFT algorithm

elif algorithm == "CGFT":
    init_prob = 0.5
    direction_probability = [init_prob] * params
    direction_probability_change_size = 0.001

    param_probability = [1.0 / params] * params
    param_probability_change_size = 0.001

    global_disc_inputs = set()
    global_disc_inputs_list = []

    local_disc_inputs = set()
    local_disc_inputs_list = []

    tot_inputs = set()

    global_iteration_limit = N
    local_iteration_limit = 2000
    model = joblib.load(classifier_name)


    def normalise_probability():
        probability_sum = 0.0
        for prob in param_probability:
            probability_sum = probability_sum + prob

        for i in range(params):
            param_probability[i] = float(param_probability[i]) / float(probability_sum)


    class LocalPerturbation(object):

        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            param_choice = np.random.choice(xrange(params), p=param_probability)
            act = [-1, 1]
            direction_choice = np.random.choice(act, p=[direction_probability[param_choice],
                                                        (1 - direction_probability[param_choice])])

            if (x[param_choice] == input_bounds[param_choice][0]) or (x[param_choice] == input_bounds[param_choice][1]):
                direction_choice = np.random.choice(act)

            x[param_choice] = x[param_choice] + (direction_choice * perturbation_unit)

            x[param_choice] = max(input_bounds[param_choice][0], x[param_choice])
            x[param_choice] = min(input_bounds[param_choice][1], x[param_choice])

            ei = evaluate_input(x)

            if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
                direction_probability[param_choice] = min(
                    direction_probability[param_choice] + (direction_probability_change_size * perturbation_unit), 1)

            elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
                direction_probability[param_choice] = max(
                    direction_probability[param_choice] - (direction_probability_change_size * perturbation_unit), 0)

            if ei:
                param_probability[param_choice] = param_probability[param_choice] + param_probability_change_size
                normalise_probability()
            else:
                param_probability[param_choice] = max(param_probability[param_choice] - param_probability_change_size,
                                                      0)
                normalise_probability()

            return x


    class GlobalDiscovery(object):
        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            for i in xrange(params):
                random.seed(time.time())
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])

            x[sensitive_param - 1] = input_bounds[sensitive_param - 1][0]
            return x


    def evaluate_input(inp):
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

        # return (abs(out0 - out1) > threshold)
        # for binary classification, we have found that the
        # following optimization function gives better results
        return abs(out1 + out0)


    def evaluate_global(inp):
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

        if abs(out0 - out1) > threshold and tuple(map(tuple, inp0)) not in global_disc_inputs:
            global_disc_inputs.add(tuple(map(tuple, inp0)))
            global_disc_inputs_list.append(inp0.tolist()[0])

        # return not abs(out0 - out1) > threshold
        # for binary classification, we have found that the
        # following optimization function gives better results
        return abs(out1 + out0)


    def evaluate_local(inp):
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

        if (abs(out0 - out1) > threshold and (tuple(map(tuple, inp0)) not in global_disc_inputs)
                and (tuple(map(tuple, inp0)) not in local_disc_inputs)):
            local_disc_inputs.add(tuple(map(tuple, inp0)))
            local_disc_inputs_list.append(inp0.tolist()[0])

        # return not abs(out0 - out1) > threshold
        # for binary classification, we have found that the
        # following optimization function gives better results
        return abs(out1 + out0)


    print "Search started"
    starting_time = time.time()
    if dataset == "CENSUS":
        initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
    elif dataset == "GERMAN":
        initial_input = [1, 4, 1, 23, 4, 4, 0, 1, 2, 20, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1]
    elif dataset == "BANK":
        initial_input = [0, 5, 1, 2, 0, 23, 0, 1, 2, 20, 5, 35, 39, 1, 1, 2]
    minimizer = {"method": "L-BFGS-B"}

    global_discovery = GlobalDiscovery()
    local_perturbation = LocalPerturbation()


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
            print len(test_suite_CT_Extra)
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

        # print "length of the set difference: "+ str(len(setDifference))
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
        evaluate_global(inp)

    seedData = len(global_disc_inputs_list)

    for input in global_disc_inputs_list:
        basinhopping(evaluate_local, input, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer,
                     niter=local_iteration_limit)

    print "Total evaluated data: " + str(len(tot_inputs))
    print "Number of seed data: " + str(seedData)
    print "Number of discriminatory data: " + str(len(global_disc_inputs_list) + len(local_disc_inputs_list))
    print "Percentage of discriminatory data: " + str(float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                                                      / float(len(tot_inputs)) * 100)
    elapsed_time = time.time() - starting_time
    print "Number of discriminatory data per second: " \
          + str((len(global_disc_inputs_list) + len(local_disc_inputs_list)) / elapsed_time)
    print ("Execution_time: {0}".format(elapsed_time) + "[sec]")

    with open("result/" + algorithm + "/" + dataset + "_" + classifier + "_" + str(N) + ".txt", "a") as myfile:
        myfile.write(str(len(tot_inputs)) + " "
                     + str(seedData) + " "
                     + str(len(global_disc_inputs_list) + len(local_disc_inputs_list)) + " "
                     + str(float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                           / float(len(tot_inputs)) * 100) + " "
                     + str((len(global_disc_inputs_list) + len(local_disc_inputs_list)) / elapsed_time) + " "
                     + "{0}".format(elapsed_time) + " "
                     + "\n"
                     )

    print "Search ended"

# RSUTT algorithm

elif algorithm == "RSUTT":
    for times in range(repeat_times):
        search_finished = False
        
        disc_inputs = set()
        disc_inputs_list = []

        tot_inputs = set()

        global_iteration_limit = N
        local_iteration_limit = 0
        local_cnt = 0

        model = joblib.load(classifier_name)


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
            
            global search_finished
            
            if search_finished:
                return
            
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
                
                if len(disc_inputs_list) >= dataset_num:         
                    search_finished = True                    

            # return not abs(out0 - out1) > threshold
            # for binary classification, we have found that the
            # following optimization function gives better results
            return abs(out1 + out0)

        # Local search algorithm for each dataset

        def my_local_search(inp):
            global local_cnt
            global search_finished
            for param in range(params):
                if search_finished:
                    break
                
                if param == sensitive_param - 1:
                    continue
                else:
                    for direction in [-1, 1]:
                        if search_finished:
                            break
                        
                        inp2 = copy.copy(inp)
                        inp2[param] = inp2[param] + direction
                        if inp2[param] < input_bounds[param][0] and direction == -1:
                            continue
                        elif inp2[param] > input_bounds[param][1] and direction == 1:
                            continue
                        elif tuple(map(tuple, np.reshape(np.asarray(inp2), (1, -1)))) in tot_inputs:
                            continue
                        evaluate_disc(inp2)
                        local_cnt += 1


        print "Search started"
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
                print len(test_suite_CT_Extra)
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

            # print "length of the set difference: "+ str(len(setDifference))
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
            if search_finished:
                break
            evaluate_disc(inp)

        seedData = len(disc_inputs_list)

        local_iteration_limit = seedData * 2000

        for input in disc_inputs_list:
            if local_cnt < local_iteration_limit and not search_finished:
                my_local_search(input)
            else:
                break

        #print "Total evaluated data: " + str(len(tot_inputs))
        #print "Number of seed data: " + str(seedData)
        #print "Number of discriminatory data: " + str(len(disc_inputs_list))
        #print "Percentage of discriminatory data: " + str(float(len(disc_inputs_list)) / float(len(tot_inputs)) * 100)
        #elapsed_time = time.time() - starting_time
        #print "Number of discriminatory data per second: " + str(len(disc_inputs_list) / elapsed_time)
        #print ("Execution_time:{0}".format(elapsed_time) + "[sec]")
        #print "Search ended"
        
        if dataset == "CENSUS":
            dataset_name = "Adult"
        elif dataset == "BANK":
            dataset_name = "Bank"
        elif dataset == "GERMAN":
            dataset_name = "Credit"
            
        # select random disc data
        if dataset_name == "Adult" or dataset_name == "Bank":
            if len(disc_inputs_list) >= 1000:
                disc_inputs_list = random.sample(disc_inputs_list, 1000)
        elif dataset_name == "Credit":
            if len(disc_inputs_list) >= 50:
                disc_inputs_list = random.sample(disc_inputs_list, 50)
        
        output_dir = "../distance_results/{}/{}/{}".format(algorithm.upper(), "{}-{}".format(dataset_name, sensitive_name), classifier)
        
        if not os.path.exists(output_dir):
            print("{}path".format(output_dir))
            raise IOError("Output directory not found: {}".format(output_dir))
        
        disc_file = os.path.join(output_dir, "{}_{}-{}_{}_{}_{}.csv".format(algorithm, dataset_name, sensitive_name, classifier, N, times))
        with open(disc_file, 'wb') as csvfile:
            import csv
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(disc_inputs_list) 
        
        print("Saving the detected discriminatory instances to: {}".format(disc_file))
        
        
        # check distance
        
        def L1_distance(disc1,disc2):
            distance = 0
            for i in range(len(disc1)):
                #print(distance)
                difference = abs(disc1[i]-disc2[i])
                if dataset_name=="Adult":
                    distance += float(difference)/(config_census.input_bounds[i][1]- config_census.input_bounds[i][0])
                elif dataset_name=="Bank":
                    distance += float(difference)/(config_bank.input_bounds[i][1]- config_bank.input_bounds[i][0])
                elif dataset_name=="Credit":
                    distance += float(difference)/(config_german.input_bounds[i][1]- config_german.input_bounds[i][0])
            return distance
        
        pairwisedistance = 0
        count=0
        disc_num = len(disc_inputs_list)
        for i in range(0,disc_num-1):
            for j in range(i+1,disc_num):
                pairwisedistance += L1_distance(disc_inputs_list[i], disc_inputs_list[j])
                count += 1
        pairwisedistance = float(pairwisedistance) / count

        
        # Save pairwise distance
        pairwise_file = os.path.join(output_dir, "pairwise_distance.txt")
        with open(pairwise_file, 'a') as f: 
            f.write("{} ({}_{}-{}_{}_{}_{})\n".format(pairwisedistance, algorithm, dataset_name, sensitive_name, classifier, N, times))
        
        print("Pairwise distance ({:.6f}) saved to: {}".format(pairwisedistance, pairwise_file))

else:
    print "The algorithm name is wrong."

