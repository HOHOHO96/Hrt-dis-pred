import numpy as np
import pandas as pd
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask , render_template,request
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('form.html')
@app.route('/name',methods=["GET","POST"])
def name():
    if request.method == "POST":
       o=[]
       a = request.form.get("a1")
       b = request.form.get("b1")
       c = request.form.get("c1")
       d = request.form.get("d1")
       e = request.form.get("e1")
       f = request.form.get("f1")
       g = request.form.get("g1")
       h = request.form.get("h1")
       i = request.form.get("i1")
       j = request.form.get("j1")
       k = request.form.get("k1")
       l = request.form.get("l1")
       m = request.form.get("m1")
       o.append(int(a))
       o.append(int(b))
       o.append(int(c))
       o.append(int(d))
       o.append(int(e))
       o.append(int(f))
       o.append(int(g))
       o.append(int(h))
       o.append(int(i))
       o.append(int(j))
       o.append(int(k))
       o.append(int(l))
       o.append(int(m))
    
       heart_data = pd.read_csv(r"C:\Users\VISHNU\Downloads")
       
       heart_data.head()
      
       heart_data.tail()
       
       heart_data.shape
      
       heart_data.info()
       
       heart_data.isnull().sum()
       
       heart_data.describe()
     
       heart_data['target'].value_counts()
       X = heart_data.drop(columns='target', axis=1)
       Y = heart_data['target']
       X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
       model = LogisticRegression()
       
       model.fit(X_train, Y_train)
      
       X_train_prediction = model.predict(X_train)
       training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
       
       X_test_prediction = model.predict(X_test)
       test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
       input_data = tuple(o)
       print(type(input_data))
       
       input_data_as_numpy_array= np.asarray(input_data)
       
       input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
       prediction = model.predict(input_data_reshaped)
       print(prediction)
       if (prediction[0]== 0):
          z="The Person does not have a Heart Disease"
          print('The Person does not have a Heart Disease')
       else:
          z="The Person does has a Heart Disease"
          print('The Person has Heart Disease')
       return z
    return render_template("form.html")
if __name__=='__main__':
    app.run()