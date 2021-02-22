import os
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def read_data(filename):
    """
    Read data from filename
    """
    df = pd.read_csv(filename)
    return df

def preprocess(test, train):
    """
    Preprocess data files
    """
    # Okay, lots of missing age values. 
    # Filling them with the mean value of the column
    train["Age"] = train["Age"].fillna(train["Age"].mean())
    test["Age"] = test["Age"].fillna(train["Age"].mean())

    # 1 missing Fare in test, filling with the mean
    test["Fare"] = test["Fare"].fillna(train["Fare"].mean())

    # sklearn does not like columns with  categorical values
    # make them binary dummy variables instead
    train = pd.get_dummies(train, columns=["Pclass", "Embarked", "Sex"])
    test =  pd.get_dummies(test, columns=["Pclass", "Embarked", "Sex"])

    # age as a binary measure
    train['Old'] = np.where(train['Age']>=50, 1, 0)
    train['Child'] = np.where(train['Age']>=18, 0, 1)
    train['Middle'] = np.where((train['Age']>=18) & (train['Age']>=50), 1, 0)
    test['Old'] = np.where(test['Age']>=50, 1, 0)
    test['Child'] = np.where(test['Age']>=18, 0, 1)
    test['Middle'] = np.where((test['Age']>=18) & (test['Age']>=50), 1, 0)

    # a measure if people have a cabin or not
    train['Cabin_0'] = train['Cabin'].isnull()*1
    test['Cabin_0'] = test['Cabin'].isnull()*1

    # fare as binary 
    train['FareBi'] = np.where(train['Fare']>=29.5, 1, 0)
    test['FareBi'] = np.where(test['Fare']>=29.5, 1, 0)

    # Removing unused columns
    uninformative_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Age", "Fare"]
    train = train.drop(columns=uninformative_cols)
    test = test.drop(columns=uninformative_cols)

    X = train.loc[:, train.columns != "Survived"]
    Y = train["Survived"]

    X_test = test.loc[:, train.columns != "Survived"]
    Y_test = test["Survived"]

    return X, Y, X_test, Y_test

def train_elastic_net(l1_ratio, X, Y):
    # Elastic net is a form regularization that combines L1 regularization
    # (Lasso) and L2 (ridge) 
    model = LogisticRegression(penalty = 'elasticnet', 
                            solver = 'saga',
                            l1_ratio = l1_ratio)
    # Make subset of training data containing everything except the label
    # Make subset containing only the label
    model.fit(X, Y)

    return model

def train_model(model_type, X, Y, **kwargs):
    """
    Fit any type of (sk-learn) model
    **kwargs allows us to us to input a dictionary containing 
    any number of parameters. For instance, if the model type is 
    LogisticRegression and you want to set the l1_ratio you can do
    train_model(LogisticRegression, X, Y, {"l1_ratio" : 0.1})
    if you wanted to set multiple parameters you would just add 
    more items to the dictionary, e.g. 
    {"l1_ratio" : 0.1, "penalty" : "elasticnet"}

    As such, this function fills the same purpose as train_elastic_net above,
    but is more general and can be used with any sk-learn model
    """
    model = model_type(**kwargs)
    model.fit(X, Y)

    return model

def train_from_dict(model_dict, X, Y, X_test, Y_test):
    """
    This function used the train_model function just defined
    to loop over a dictionary of models and their corresponding parameters
    See how it is used in action in the bottom of the script

    Given a dictionary of form
    {name : {model_function : {param_key : param_ value}}}
    Train each model and return performance
    """
    
    performance = {}
    for name in model_dict.keys():
            # Extract the model function
            model_type = list(model_dict[name].keys())[0]
            print(f"""Training {name} with parameters:
                        {model_dict[name][model_type]}""")

            trained_model = train_model(model_type, 
                                        X, Y, 
                                        **model_dict[name][model_type])
            
            acc_train, acc_test = get_performance(trained_model, 
                                                  X, Y, 
                                                  X_test, Y_test)
        
            performance[name] = {"acc_train" : acc_train, 
                                "acc_test" : acc_test}
    
    return performance

def get_performance(model, X, Y, X_test, Y_test):

    # Fit model on training data
    # See how well the model does on the training data
    yhat = model.predict(X)
    acc_train = accuracy_score(Y, yhat)
    print(f"Accuracy on train data: {acc_train}")
    #print(confusion_matrix(Y, yhat))

    # Test the model on the testing set
    yhat_test = model.predict(X_test)
    acc_test = accuracy_score(Y_test, yhat_test)
    print(f"Accuracy on test data: {acc_test}")
    #print(confusion_matrix(Y_test, yhat_test))

    return acc_train, acc_test

if __name__ == '__main__':
    # Run script from a terminal with
    # python w3_live_code_end.py

    data_folder = "./Class2/titanic" # set to the name of the folder where you keep the data
    
    model_dict = {"SVC_0.1" : {LinearSVC : 
                                        {"C" : 0.1}},
                    "SVC_1" : {LinearSVC : 
                                        {"C" : 1}},
                    "Logistic" : {LogisticRegression : 
                                        {"penalty" : "elasticnet",
                                        "l1_ratio" : 0.5,
                                        "solver" : "saga"
                                        }},
                    "AdaBoost" : {AdaBoostClassifier : 
                                        {"n_estimators" : 50,
                                        "learning_rate" : 1
                                        }},
                    "RandomForest" : {RandomForestClassifier : 
                                        {"n_estimators" : 3
                                        }},
                    "ComplementNB" : {ComplementNB : 
                                        {"alpha" : 1.0
                                        }}
    }

    train_sets = glob.glob(data_folder + "/train*")

    performance = {}
    for dataset in train_sets:
        # Loop through each data partition and run pipeline
        print(f"\nTraining and testing on {dataset}\n")
        test = read_data(os.path.join(data_folder, "test.csv"))
        train = read_data(dataset)
        X, Y, X_test, Y_test = preprocess(test, train)

        # Method 1: using train_elastic_net
        #
        # for l1_ratio in [0, 0.5, 1]:
        #     print(f"Training with l1 ratio of {str(l1_ratio)}")
        #     model = train_elastic_net(l1_ratio=l1_ratio, X=X, Y=Y)
        #     acc_train, acc_test = get_performance(model, X, Y, X_test, Y_test)

        # Method 2: using train_model
        # model = train_model(LinearSVC, X, Y, C=1)
        # acc_train, acc_test = get_performance(model, X, Y, X_test, Y_test)

        # Method 3: using train_from_dict
        performance[dataset] = train_from_dict(model_dict,
                                               X, Y, 
                                               X_test, Y_test)
       

    # Turn the results into a dataframe and save
    df = pd.DataFrame.from_dict({(i,j): performance[i][j] 
                           for i in performance.keys() 
                           for j in performance[i].keys()},
                       orient='index').\
                  reset_index()
    # df.to_csv("out.csv", index=False)
    print(df)