from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('lung_cancer_examples.csv')

# Data preprocessing
data.drop(columns=['Name', 'Surname'], inplace=True)
X = data.drop(columns=['Result'])
Y = data['Result']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=9)

# Train models
logreg = LogisticRegression(C=10)
logreg.fit(X_train, Y_train)

nbcla = GaussianNB()
nbcla.fit(X_train, Y_train)

dt = DecisionTreeClassifier(random_state=3)
dt.fit(X_train, Y_train)

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, Y_train)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['Age'])
        smokes = int(request.form['Smokes'])
        areaq = int(request.form['AreaQ'])
        alkhol = int(request.form['Alkhol'])
        
        # Predictions
        prediction_logreg = logreg.predict([[age, smokes, areaq, alkhol]])
        if prediction_logreg == 1:
            prediction_logreg = 'Patient is diagnosed with Lung Cancer'
        else:
            prediction_logreg = 'Patient is not diagnosed with Lung Cancer'
              
        prediction_nbcla = nbcla.predict([[age, smokes, areaq, alkhol]])
        if prediction_nbcla == 1:
            prediction_nbcla = 'Patient is diagnosed with Lung Cancer'
        else:
            prediction_nbcla = 'Patient is not diagnosed with Lung Cancer'
             
        prediction_dt = dt.predict([[age, smokes, areaq, alkhol]])
        if prediction_dt == 1:
            prediction_dt = 'Patient is diagnosed with Lung Cancer'
        else:
            prediction_dt = 'Patient is not diagnosed with Lung Cancer' 
            
        prediction_knn = knn.predict([[age, smokes, areaq, alkhol]])
        if prediction_knn == 1:
            prediction_knn = 'Patient is diagnosed with Lung Cancer'
        else:
            prediction_knn = 'Patient is not diagnosed with Lung Cancer' 
        
        return render_template('result.html', 
                               prediction_logreg=prediction_logreg,
                               prediction_nbcla=prediction_nbcla,
                               prediction_dt=prediction_dt,
                               prediction_knn=prediction_knn)

if __name__ == '__main__':
    app.run(debug=True)
