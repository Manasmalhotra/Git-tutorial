from Deployy import*

model=loan_model('model')
model.load_clean('train_1.csv')
y_pred=model.predicted_output()
print(y_pred)
results=model.predicted_results()
print(results)
