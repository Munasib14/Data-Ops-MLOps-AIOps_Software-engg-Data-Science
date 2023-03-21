import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts, log_metrics


## Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Starting the experiment')
    
    ## mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name = 'mlflow titanic')

    ## read the data
    titanic_data = pd.read_csv('Titanic+Data+Set.csv')

    ## Basic EDA

    log_param("Befor Value counts", titanic_data['Survived'].value_counts())

    titanic_data = titanic_data.drop(columns = ['Cabin','PassengerId','Name','Ticket',],axis=1)
    titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())

    le = LabelEncoder()
    titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'])
    titanic_data['Embarked'] = le.fit_transform(titanic_data['Embarked'])
    titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
    titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}})
    print(titanic_data['Survived'].value_counts())
    
    log_param("After Value counts", titanic_data['Survived'].value_counts())

    ## splitting data into training and test set for independent attributes 75% to train and 25% to test
    X_train, X_test, y_train, y_test = train_test_split(titanic_data.drop('Survived', axis=1), titanic_data['Survived'],
                                                        test_size=.25,
                                                        random_state=22)
    print(X_train.shape, X_test.shape)
    log_param("Train shape",X_train.shape )
    log_param("Test shape",X_test.shape )

    model_entropy = DecisionTreeClassifier(criterion = "entropy",
                               max_depth=10, min_samples_leaf=5)

    model_entropy.fit(X_train, y_train)
    print("Model trained")

    train_accuracy = model_entropy.score(X_train, y_train)  ## performance on train data
    test_accuracy = model_entropy.score(X_test, y_test)  ## performance on test data

    log_metric("Accuracy for this run", test_accuracy)
    pred_test = model_entropy.predict(X_test)
    pref_metrics = {"precison_test" : precision_score(y_test, pred_test, average='micro'),
                    "recall_test" : recall_score(y_test, pred_test, average='micro'),
                    "f1_score_test" : f1_score(y_test, pred_test, average='micro')
                    }
    log_metrics(pref_metrics)
    ## log_metric("Accuracy for this run", test_accuracy)
    mlflow.sklearn.log_model(model_entropy, "DecisionTreeModel")
    ## mlflow.log_artifact('winequality-red.csv')