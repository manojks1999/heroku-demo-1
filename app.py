from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import math

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("laptop_pred.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    output=math.floor(prediction[0])

    print(output)

    if output:
        return render_template('results.html',pred='You can buy this laptop at {} Rs '.format(str(output)))


if __name__ == '__main__':
    app.run(debug=True)
