import random
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold



random.seed(1)

RF_param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 5, 10],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}


SVM_param_grid = {

}



def ML_pipeline(PCA_X_train, y_train, PCA_X_test, y_test):
    
    
    # create stratified splits
    skf = StratifiedKFold(n_splits=5, random_state=46) 

    # find best params
    SVM_best_params = SVM_finetune(skf, PCA_X_train, y_train, PCA_X_test, y_test)
    RF_best_params = RF_finetune(skf, PCA_X_train, y_train, PCA_X_test, y_test)
    
    # train best model
    SVM_model = SVC(**SVM_best_params)
    RF_model = RandomForestClassifier(**RF_best_params)
    
    # evaluate model
    model_evaluate(SVM_model, PCA_X_train, y_train, PCA_X_test, y_test)
    model_evaluate(RF_model, PCA_X_train, y_train, PCA_X_test, y_test)
    


def SVM_finetune():
    return







def SVM_train(PCA_X_train, y_train, PCA_X_test, y_test):
    # Create an SVM classifier
    svm_classifier = SVC()

    # Train the classifier
    svm_classifier.fit(PCA_X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_classifier.predict(PCA_X_test)

    # Evaluate the model
    classification_metrics = classification_report(y_test, y_pred,zero_division=1)
    print("Classification Report:")
    print(classification_metrics)


def RF_finetune(PCA_X_train, y_train):
    
    rf_classifier = RandomForestClassifier()
    grid_search = GridSearchCV(rf_classifier, RF_param_grid, cv=5, scoring='accuracy')
    grid_search.fit(PCA_X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    return best_params



def model_evaluate(model,params,X_train, y_train, X_test, y_test):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    classification_metrics = classification_report(y_test, y_pred,zero_division=1)
    print("Classification Report:")
    print(classification_metrics)
