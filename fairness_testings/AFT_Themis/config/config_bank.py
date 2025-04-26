# ORIGINAL FILE
params = 16

sensitive_param_age = 1     # Starts at 1.

input_bounds = []

input_bounds.append([1, 9])     #age (Discriminatory parameter)
input_bounds.append([0, 11])
input_bounds.append([0, 2])
input_bounds.append([0, 3])
input_bounds.append([0, 1])
input_bounds.append([-20, 179])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 2])
input_bounds.append([1, 31])
input_bounds.append([0, 11])
input_bounds.append([0, 99])
input_bounds.append([1, 63])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 3])

threshold = 0

perturbation_unit = 1

retraining_inputs = "Retrain_Example_File.txt"