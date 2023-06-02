import os
import random
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from save_load_utils import load_model, save_model


random.seed(1)

RF_param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 5, 10],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}


SVM_param_grid = {
    'C' : [0.1,1,10],
    'kernel' : ['rbf','poly','sigmoid'],
    'gamma' : ['scale','auto']
}



def ML_pipeline(PCA_X_train, y_train, PCA_X_test, y_test, SVM_FILE, RF_FILE):
    

    if os.path.exists(SVM_FILE) and os.path.exists(RF_FILE):
        SVM_model = load_model(SVM_FILE)
        RF_model = load_model(RF_FILE)
    else:
        # create stratified splits
        skf = StratifiedKFold(n_splits=5, random_state=46, shuffle=True) 

        # find best params
        SVM_best_params = SVM_finetune(skf, PCA_X_train, y_train)
        RF_best_params = RF_finetune(skf, PCA_X_train, y_train)
        
        # train best model
        SVM_model = SVC(**SVM_best_params)
        SVM_model.fit(PCA_X_train,y_train.ravel())

        RF_model = RandomForestClassifier(**RF_best_params)
        RF_model.fit(PCA_X_train,y_train.ravel())

        # save models
        save_model(SVM_model,SVM_FILE)
        save_model(RF_model,RF_FILE)
    
    # evaluate model
    model_evaluate(SVM_model, PCA_X_train, y_train, PCA_X_test, y_test)
    model_evaluate(RF_model, PCA_X_train, y_train, PCA_X_test, y_test)




def SVM_finetune(skf, PCA_X_train, y_train):
    print('Starting GridSearch in SVM model')
    svm_classifier = SVC()
    grid_search = GridSearchCV(svm_classifier, SVM_param_grid, cv=skf, scoring='accuracy')
    grid_search.fit(PCA_X_train, y_train.ravel())

    best_params = grid_search.best_params_
    print('SVM best params : ',best_params)
    best_score = grid_search.best_score_
    return best_params


def RF_finetune(skf, PCA_X_train, y_train):
    print('Starting GridSearch in Random Forest model')
    rf_classifier = RandomForestClassifier()
    grid_search = GridSearchCV(rf_classifier, RF_param_grid, cv=skf, scoring='accuracy')
    grid_search.fit(PCA_X_train, y_train.ravel())

    best_params = grid_search.best_params_
    print('RF best params : ',best_params)
    best_score = grid_search.best_score_
    return best_params



def model_evaluate(model,X_train, y_train, X_test, y_test):
    model.fit(X_train,y_train.ravel())
    y_pred=model.predict(X_test)
    classification_metrics = classification_report(y_test, y_pred,zero_division=1)
    print("Classification Report:")
    print(classification_metrics)
