import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# mydata = pd.read_csv(r'C:\Users\SKH-PC02\Downloads\IschemicHeartDiseasePy\IschemicHeartDisease.csv')
mydata = pd.read_csv(r'C:\Users\Toei\Desktop\IschemicHeartDiseasePy\IschemicHeartDisease.csv')

x = mydata[['ChestTightness','ChestPain','Smart','HeartPalpitations','Squeamish','Faint','Gasp','Tired','Choking','EpigastricCongestion']]
y = mydata['Ischemicheart']

x_train, x_test, y_traain, y_test = train_test_split(x,y, test_size=0.2, random_state=1)

modelRF = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
modelRF.fit(x_train, y_train)

y_pred = modelRF.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import pickle
filename = 'IschemicHeartDisease.sav'
pickle.dump(modelRF,open(filename, 'wb'))

load_model = pickle.load(open(filename,'rb'))

feature_names = ['ChestTightness','ChestPain','Smart','HeartPalpitations','Squeamish','Faint','Gasp','Tired','Choking','EpigastricCongestion']

# Convert input to a DataFrame
input_data = pd.DataFrame([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], columns=feature_names)

# Make prediction
print(load_model.predict(input_data))