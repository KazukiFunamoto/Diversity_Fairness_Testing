# Create models
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

dataset = sys.argv[1]
num = int(sys.argv[2])

# Reading the dataset
if dataset == "CENSUS":
    df = pd.read_csv('datasets/census.csv')

elif dataset == "GERMAN":
    df = pd.read_csv('datasets/german.csv')

elif dataset == "BANK":
    df = pd.read_csv('datasets/bank.csv')

else:
    print "The dataset name is wrong."

data = df.values

X = data[:, :-1]
Y = data[:, -1]

def DT():
    model = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                                   max_features=None, max_leaf_nodes=None,
                                   min_impurity_decrease=0.0, min_impurity_split=None,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0,
                                   random_state=42, splitter='best')

    # Fitting the model with the dataset
    model = model.fit(X, Y)

    if dataset == "CENSUS":
        pd.to_pickle(model, 'census/DT_CENSUS.pkl')

    elif dataset == "GERMAN":
        pd.to_pickle(model, 'german/DT_GERMAN.pkl')

    elif dataset == "BANK":
        pd.to_pickle(model, 'bank/DT_BANK.pkl')

    return model


def MLPC():
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                          alpha=0.0001, batch_size='auto', learning_rate='constant',
                          learning_rate_init=0.001, power_t=0.5, max_iter=200,
                          shuffle=True, random_state=42, tol=0.0001,
                          verbose=False, warm_start=False, momentum=0.9,
                          nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                          beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = model.fit(X, Y)

    if dataset == "CENSUS":
        pd.to_pickle(model, 'census/MLPC_CENSUS.pkl')

    elif dataset == "GERMAN":
        pd.to_pickle(model, 'german/MLPC_GERMAN.pkl')

    elif dataset == "BANK":
        pd.to_pickle(model, 'bank/MLPC_BANK.pkl')

    return model


def RF():
    
    model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                   max_depth=5, max_features='auto', max_leaf_nodes=None,
                                   min_impurity_decrease=0.0, min_impurity_split=None,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                                   oob_score=False, random_state=42, verbose=0,
                                   warm_start=False)
    
    #model = RandomForestClassifier()

    # Fitting the model with the dataset
    model = model.fit(X, Y)

    if dataset == "CENSUS":
        pd.to_pickle(model, 'census/RF_CENSUS.pkl')

    elif dataset == "GERMAN":
        pd.to_pickle(model, 'german/RF_GERMAN.pkl')

    elif dataset == "BANK":
        pd.to_pickle(model, 'bank/RF_BANK.pkl')

    return model

if num == 1:
    DT()
    MLPC()
    RF()
    print "classifier"

else:
    print "not classifier"
