
# NOTE: what is left to add is doing all the combinations of my features to compute miltivariate statistics

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn imports
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

# get our data from difference location
import sys
sys.path.append("/Users/migashane/CodeUp/Data_Science/classification-exercises")
import prepare

# for funtion annotations
from typing import Union
from typing import Tuple

# separate features form target
def feature_separation_(train: pd.DataFrame, validate: pd.DataFrame, test: pd.DataFrame,
                    feature_cols: list, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Goal: separate the features from target varable
    paremeters:
        train: training data
        validate: validation data
        test: testing data

    return:
        x_train: separated training features
        y_train: target traing variable
        x_validate: separated validation features
        y_validate: target validatetion variable
        x_test: separated testing features
        y_test: target testing variable
    """
    # training separation
    x_train = train[feature_cols]
    y_train = train[target_col]

    # validation separation
    x_validate = validate[feature_cols]
    y_validate = validate[target_col]

    # test separation
    x_test = test[feature_cols]
    y_test = test[target_col]

    return x_train, y_train, x_validate, y_validate, x_test, y_test

# train the model on the training data
def train_model_(x_train: pd.DataFrame, y_train: pd.Series, num_neighbors:int) -> object:
    """"
    Goal: create a train the model
    parameters:
        x_train: features data
        y_train: target variable
    return:
        knn_estimator: The knn model estimator object

    """
    # create a knn object
    #                          n_neighborsint(default=5) 
    knn = KNeighborsClassifier(n_neighbors= num_neighbors, weights='distance', p=1)
    #                                                        p=1 uses the manhattan distance

    # fit training data to the object
    knn_estimator = knn.fit(x_train, y_train)

    return knn_estimator

# make predictions
def make_predictions_(estimator_obj: object, xTrain: pd.DataFrame, xVal: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Goal: make predictions
    parameters:
        estimator_obj: estimator object for when model was fit
        xTrain: x_train features to make predictions from
        xVal: x_validate features to make predictions from

    return:
        y_pred: model predictions
        y_pred_proba: model prediction probability (how model decided)
        pred_classes: univer values the model make choices from
    """
    y_pred = estimator_obj.predict(xTrain)
    y_pred_proba = estimator_obj.predict_proba(xTrain)
    pred_classes = estimator_obj.classes_

    # validatation prediction
    y_Val_Pred = estimator_obj.predict(xVal)
    y_Val_Pred_proba = estimator_obj.predict_proba(xVal)

    return y_pred, y_pred_proba, y_Val_Pred, y_Val_Pred_proba,pred_classes

# evaluate a single model
def evaluate_model_(estimator_obj: object, xTrain: pd.DataFrame, yTrain: pd.Series,
                    xVal: pd.DataFrame, yVal: pd.Series,yPred: pd.Series, yValPred:pd.Series) -> Tuple[float, float]:
    """
    Goal: validate the model

    return:
        train_score: accuracy score of the training dat
        validate_score: accuracy score of the validation data
    """
    train_score= estimator_obj.score(xTrain, yTrain)
    validate_score = estimator_obj.score(xVal, yVal)

    # confusion matrix agaist the prediction
    train_confussion_matrix = confusion_matrix(yTrain, yPred)
    validate_confussion_matrix = confusion_matrix(yVal, yValPred)

    return train_score, validate_score, train_confussion_matrix, validate_confussion_matrix

def computed_models_dataframe_(trainAccuracy: float, valAccuracy: float, baseLineAccuracy: float, model_num: int) -> dict:
    """
    Goal: add mode accuracy differences in likke like object that will be tranformed in a final dataframe
    """
    # create a dictionary of scores
    model_info = {
        "num_neighbors": model_num,
        "train_score": trainAccuracy,
        "validate_score": valAccuracy,
        "difference": trainAccuracy - valAccuracy,
        "train_baseline_diff": baseLineAccuracy - trainAccuracy,
        "val_baseline_diff": baseLineAccuracy - valAccuracy
    }

    return model_info

def create_visuals_(df: pd.DataFrame) -> Tuple[plt.plot, plt.plot, plt.plot]:
    # PLOT train_vs_validation scores
    df[df.columns[:-3]].set_index("num_neighbors").plot()
    plt.ylabel('accuracy score')
    plt.xticks(np.arange(2,20))
    plt.grid()
    train_vs_validation = plt.show()

    # distance_from_baseline
    df.drop(columns=["difference","validate_score", "train_score"]).set_index("k").plot()
    plt.ylabel('Distance from base line')
    plt.xticks(np.arange(2,20))
    plt.grid()
    distance_from_baseline = plt.show()

    # distance_from_each_other
    df[["difference", "k"]].set_index("k").plot()
    plt.ylabel('Distance from each other')
    plt.xticks(np.arange(2,20))
    plt.grid()
    distance_from_each_other = plt.show()

    return (train_vs_validation, distance_from_baseline, distance_from_each_other)

# Perfo4m full knn modeling
def model_knn_(train: pd.DataFrame, validate: pd.DataFrame, test: pd.DataFrame,
              feature_col: list, target_col: str, baseline_col: str,
              numer_of_models = 1) -> Tuple[Union[Union[pd.Series, np.array], 
                                               Union[float, float],
                                               Union[np.array, np.array],
                                               Union[str,str, object]], pd.DataFrame]:
    """
    Goal: Compute K-nearest neighbers model and return the result of the model(s)
    paremeters:

    return:
        model_item_full_discription: This is a dictionary of dictionaries that contain the floowing information tuples
            1:{"predictions": (yPred,yPred_proba), 
              "accuracy_scores":(trainAccuracy, valAccuracy), 
              "confusion_metrices":(train_confussion_matrix, validate_confussion_matrix),
              "classification_reports":(train_class_report,val_class_report), 
              "model_object":knn_estimator_obj}
    """
    
    model_item_full_discription = {}
    model_information_dataframe = []

    # compute base line accuracy score
    baseLineAccuracy = accuracy_score(train[target_col],train[baseline_col])

    for model in range(1, numer_of_models + 1):
        # for iteration in range(1, numer_of_models + 1):
            # step 1: separate features from target
        xTrain, yTrain, xVal, yVal, xtest, yTest = feature_separation_(train, validate, test, 
                                                                    feature_col, target_col)
        # step 2: knn estimator object
        knn_estimator_obj = train_model_(xTrain, yTrain, model)

        # step 3: make predictions
        yPred, yPred_proba, yVal_pred, yVal_proba, predClasses = make_predictions_(knn_estimator_obj, xTrain, xVal)
        
        # # step 4: evaluate model
        trainAccuracy, valAccuracy, train_confussion_matrix, validate_confussion_matrix = evaluate_model_(knn_estimator_obj, xTrain, yTrain, xVal, yVal, yPred, yVal_pred)

        # classification replort
        train_class_report = classification_report(yTrain, yPred)
        val_class_report = classification_report(yVal, yVal_pred)

        # create list of models
        modelInfo = computed_models_dataframe_(trainAccuracy, valAccuracy, baseLineAccuracy, model)
        model_information_dataframe.append(modelInfo)

        # to facilitate unpackikn go varables I am using tuples in the dictionary
        # these will all be returned to the user
        model_return = {
            "predictions": (yPred,yPred_proba),
            "accuracy_scores": (trainAccuracy, valAccuracy),
            "confusion_metrices": (train_confussion_matrix, validate_confussion_matrix),
            "classification_reports": (train_class_report,val_class_report),
            "model_object": knn_estimator_obj,

        }

        # add full discripion of model to be returned to the user
        model_item_full_discription[model] = model_return

    # get visuals
    accuracy_differences_df = pd.DataFrame(model_information_dataframe)
    models_visuals = create_visuals_(accuracy_differences_df)

    return model_item_full_discription, accuracy_differences_df, models_visuals
