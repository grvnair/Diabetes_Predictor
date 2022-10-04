import requests

url = 'http://127.0.0.1:5000/predict_api'

r = requests.post(url, json={'Pregnancies':5 ,'Glucose':166 ,'BloodPressure':72 ,'SkinThickness':19 ,'Insulin':175 ,
                  'BMI':25.8 ,'DiabetesPedigreeFunction':0.587 ,'Age':51})

print(r.json())