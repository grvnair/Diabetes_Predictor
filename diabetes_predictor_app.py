import numpy as np
from flask import Flask, jsonify, request, render_template
from sklearn.preprocessing import StandardScaler
import pickle


app = Flask(__name__)
model_dict = pickle.load(open('diabetes_model.sav', 'rb'))
print(model_dict)
model = model_dict["classifier"]
scaler = model_dict["scaler"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    input_data_to_array = np.asarray(int_features)
    reshaped_input_data = input_data_to_array.reshape(1, -1)
    standardized_input_data = scaler.transform(reshaped_input_data)
    # final_features =[np.array(standardized_input_data)]
    prediction = model.predict(standardized_input_data)

    output = prediction[0]

    if prediction[0] == 0:
        return f"The person is non diabetic! :)"
    else:
        return f"The person is diabetic :("

@app.route('/predict_api', methods = ['POST'])
def predict_api():

    data = request.get_json(force=True)
    print(data)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    if prediction[0] == 0:
        return f"The person is non diabetic! :) \n\n{data} Gaurav Nair \n\n{prediction}"
    else:
        return f"The person is diabetic :( \n\n{data} Gaurav Nair \n\n{prediction}"
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)