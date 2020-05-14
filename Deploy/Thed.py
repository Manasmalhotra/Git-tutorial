import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

###Read data#####################
data=pd.read_csv('train_1.csv')
##########Correct variables##########################
data.loc[data['Dependents']=="3+",'Dependents']=3
print(data['Dependents'].value_counts())
data['Total']=data['ApplicantIncome']+data['CoapplicantIncome']
data.drop(['ApplicantIncome','CoapplicantIncome'],1,inplace=True)
data.to_csv('withmiss.csv',index=False)
#################Fill Misssing values#############################
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0],inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].median(),inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0],inplace=True)
data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)
data['Married'].fillna(data['Married'].mode()[0],inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace=True)
print(data.isnull().sum())
data.to_csv('clean.csv',index=False)
######################Get dummies###########################33
data=pd.get_dummies(data,columns=['Gender','Education','Property_Area','Married','Self_Employed','Loan_Status'],drop_first=True)
data.to_csv('Final.csv')
######################################################################Model#################################################
w=data['Loan_ID']
x=data.drop(['Loan_Status_Y','Loan_ID'],1)
y=data['Loan_Status_Y']
x.to_csv('x.csv',index=False)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

r=RandomForestClassifier()
r.fit(x,y)
print(r.score(x_test,y_test))

import pickle

with open('model','wb') as file:
    pickle.dump(r,file)
pickle.dump(r,open('model.pkl','wb'))

