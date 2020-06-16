import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  float_features=[float(x) for x in request.form.values()]
  final_features=np.array(float_features)
  lyst=final_features.reshape(1,16)
  prediction=model.predict(lyst)

  if prediction[0]==0:
      output="worn"
  else:
      output="not worn"

  return render_template('index.html', prediction_text= 'Your Tool is {}. Go back to predict again'.format(output))

if __name__=="__main__":
  app.run(debug=True)

