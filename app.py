from flask import Flask, render_template, request
import lung_cancer_prediction2 # Import your lung cancer prediction module

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['Age'])
        smokes = int(request.form['Smokes'])
        areaQ = int(request.form['AreaQ'])
        alkhol = int(request.form['Alkhol'])
        # Get other attributes from the form (example: smoking history)
        

        # Call your lung cancer prediction function with the input data
        prediction = lung_cancer_prediction2.predict(age, smokes, areaQ, alkhol)

        # Display prediction result
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
