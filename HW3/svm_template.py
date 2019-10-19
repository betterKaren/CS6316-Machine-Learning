# Starting code for UVA CS 4501 ML- SVM

import numpy as np
np.random.seed(37)
import random

from sklearn.svm import SVC
# Att: You're not allowed to use modules other than SVC in sklearn, i.e., model_selection.

col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
             'hours-per-week', 'native-country']
col_names_y = ['label']

numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                  'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'native-country']

# 1. Data pre-processing.
# Hint: Feel free to use some existing libraries for easier data pre-processing.
def load_data(csv_file_path):
    # your code here
    return x, y   

# 2. Select the best model with cross validation.
# Attention: Write your own hyper-parameter candidates.
def train_and_select_model(training_csv):
    x_train, y_train = load_data(training_csv)
    param_set = [
                 {'kernel': 'rbf', 'C': 1, 'degree': 1},
                 {'kernel': 'rbf', 'C': 1, 'degree': 3},
                 {'kernel': 'rbf', 'C': 1, 'degree': 5},
                 {'kernel': 'rbf', 'C': 1, 'degree': 7},
    ]
    # your code here
    return best_model, best_score

def predict(test_csv, trained_model):
    x_test, _ = load_data(test_csv)
    predictions = trained_model.predict(x_test)
    return predictions

# 3. Upload your Python code, the predictions.txt as well as a report to Collab.
# Hint: Don't archive the files or change the file names for the automated grading.
def output_results(predictions):
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            if pred == 0:
                f.write('<=50K\n')
            else:
                f.write('>50K\n')

if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    trained_model, cv_score = train_and_select_model(training_csv)
    print "The best model was scored %.2f" % cv_score
    predictions = predict(testing_csv, trained_model)
    output_results(predictions)