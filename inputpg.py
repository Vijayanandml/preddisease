import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__, template_folder=r"C:\Users\Lenovo\Documents\BREAST_CANCER_CLASSIFICATION")

df = pd.read_csv(r"C:\Users\Lenovo\Documents\BREAST_CANCER_CLASSIFICATION\data.csv")
df = df.dropna(axis=1)

labelencoder_Y = LabelEncoder()
df['diagnosis'] = labelencoder_Y.fit_transform(df['diagnosis'])

X = df.iloc[:, 2:].values  # Exclude the first column (ID) from the features
Y = df.iloc[:, 1].values

sc = StandardScaler()
X = sc.fit_transform(X)

def models(X, Y):
    models_dict = {}
    
    log = LogisticRegression(random_state=0, max_iter=1000)
    log.fit(X, Y)
    models_dict['Logistic Regression'] = log

    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X, Y)
    models_dict['Decision Tree'] = tree

    forest = RandomForestClassifier(criterion='entropy', random_state=0)
    forest.fit(X, Y)
    models_dict['Random Forest'] = forest

    kneighbors = KNeighborsClassifier()
    kneighbors.fit(X, Y)
    models_dict['K-Nearest Neighbors'] = kneighbors

    return models_dict

model = models(X, Y)

def preprocess_input(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    input_array = sc.transform(input_array)
    return input_array

def make_prediction(input_data):
    input_array = preprocess_input(input_data)
    predictions = {model_name: model.predict(input_array) for model_name, model in model.items()}  # Corrected attribute name
    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.values()  # Retrieve the input values from the HTML form
    input_data = [float(value) for value in input_data if value.strip()]  # Convert non-empty input values to floats
    
    if len(input_data) < 30:  # Assuming there are a total of 30 input fields
        error_message = "Please fill in all the input fields."
        return render_template('index.html', error_message=error_message)  # Render the error message in the template
    
    prediction = make_prediction(input_data)  # Make the prediction using the input data

    return render_template('index.html', prediction=prediction)  # Pass the prediction result to the template
if __name__ == '__main__':
    app.run(debug=True)
