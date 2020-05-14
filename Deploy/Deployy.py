import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle
      
class loan_model():
    def __init__(self, model_file):
              with open('model','rb') as model_file:
                self.reg = pickle.load(model_file)
                self.data = None
                self.results=pd.DataFrame()
    
    def load_clean(self,data_file):
        data=pd.read_csv(data_file,delimiter=',')
        data.loc[data['Dependents']=="3+",'Dependents']=3
        print(data['Dependents'].value_counts())
        data['Total']=data['ApplicantIncome']+data['CoapplicantIncome']
        data.drop(['ApplicantIncome','CoapplicantIncome'],1,inplace=True)
        data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace=True)
        data['Credit_History'].fillna(data['Credit_History'].mode()[0],inplace=True)
        data['LoanAmount'].fillna(data['LoanAmount'].median(),inplace=True)
        data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0],inplace=True)
        data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)
        data['Married'].fillna(data['Married'].mode()[0],inplace=True)
        data['Dependents'].fillna(data['Dependents'].mode()[0],inplace=True)
        print(data.isnull().sum())
        data=pd.get_dummies(data,columns=['Gender','Education','Property_Area','Married','Self_Employed','Loan_Status'],drop_first=True)
        self.data=data

    def predicted_output(self):
            if (self.data is not None):
                w=self.data.drop(['Loan_ID','Loan_Status_Y'],1)
                pred_outputs = self.reg.predict(w)
                return pred_outputs

    def predicted_results(self):
        if (self.data is not None):
                w=self.data.drop(['Loan_ID','Loan_Status_Y'],1)
                self.results['Probability'] = self.reg.predict_proba(w)[:,1]
                self.results['Prediction'] = self.reg.predict(w)
                return self.results
