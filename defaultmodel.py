import numpy as np
import pandas as pd                                 
from flask import Flask, request, jsonify, render_template
import pickle

from sklearn.base import ClassifierMixin
def train_model(data):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    data=pd.read_csv("D:\ml\Default_Fin.csv")
    x=data.iloc[:,[1,2,3]]
    y=data.iloc[:,4]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
    classifier=RandomForestClassifier(n_estimators=10,criterion='entropy')
    classifier.fit(x_train,y_train)

    return classifier

data=pd.read_csv("Default_Fin.csv")
model = train_model(data)
# Save the trained model to a file
with open('loandef.pkl','wb') as f:
    pickle.dump(model, f)

# Load the trained model from the file
model = pickle.load(open('loandef.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('user.html')
@app.route('/predict', methods=['POST'])
def predict():
     employed = int(request.form['employed'])
     balance = float(request.form['balance'])
     salary = float(request.form['salary'])

        # Create a DataFrame from the user input
     user_df = pd.DataFrame({'Employed': [employed], 'Bank Balance': [balance], 'Annual Salary': [salary]})

        # Use the trained classifier to predict the outcome
     prediction = model.predict(user_df)
     output=prediction[0]
     if output==0:
         return render_template('result.html', prediction_text='he doesnot default')
     else:
          return render_template('result1.html', prediction_text='he may default')
         
if __name__ == "__main__":
    app.run(debug=True)


       
        

    
