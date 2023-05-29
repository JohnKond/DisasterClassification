from sklearn.svm import SVC
from sklearn.metrics import classification_report



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