import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Starting the experiment')
    
    ##mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name = 'mlflow Optimized titanic model')

    mlflow.autolog()  # recored automatically

    # read the data
    titanic_data = pd.read_csv('Titanic+Data+Set.csv')


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


    X_train, X_test, y_train, y_test = train_test_split(titanic_data.drop('Survived', axis=1), titanic_data['Survived'],
                                                        test_size=.25,
                                                        random_state=22)
    print(X_train.shape, X_test.shape)
    log_param("Train shape",X_train.shape )
    log_param("Test shape",X_test.shape )


    ## Hyperparameter tuning
    model = RandomForestClassifier()

    ## Parameter Search shape
    params = [{'criterion': ['entropy', 'gini'],
                'n_estimators': [10,30,50,70,90],
                'max_features': ['sqrt', 'log'],
                'max_depth': [10,20,50],
                'min_samples_split': [2,5,8,11],
                'min_samples_leaf': [1,5,9],
                'max_leaf_nodes': [2,5,8,11]}]
    
    ## Cross validation
    cv = StratifiedKFold(n_splits=3, shuffle=True)

    ## GridSearch
    tuning = GridSearchCV(estimator=model, cv=cv, scoring='accuracy', param_grid=params)

    ## Train and optimize the estimator
    tuning.fit(X_train, y_train)

    ## Best parameters founds
    print('Best Parametyers found using GridSearch:', tuning.best_params_)

    y_predict_gs = tuning.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict_gs)

    ## Logging metrics
    log_metric("Accuracy for this run ", accuracy)

    #  logging model
    mlflow.sklearn.log_model(model, "Model")

    ## Printing the run id 
    print(mlflow.active_run().info.run_uuid)
    print("Ending the experiment")