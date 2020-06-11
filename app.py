import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from flask_bootstrap import Bootstrap
app = Flask(__name__)
Bootstrap(app)
model = pickle.load(open('model.pkl', 'rb'))
vect=pickle.load(open('vectorizer.pickle','rb'))
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    namequery = request.form['namequery']
    data = [namequery]
    vect1 = vect.transform(data).toarray()
    my_prediction = model.predict(vect1)
    return render_template('result.html',prediction = my_prediction)

if __name__ == "__main__":
    app.run(debug=True)
