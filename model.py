import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

data = pd.read_csv(
 "Social_Network_Ads.csv")


predictors = data.iloc[:,2:4]

outcome = data['Purchased']


from imblearn.combine import SMOTEENN, SMOTETomek


smote_enn = SMOTEENN(random_state=1001)

predictors_bal,outcome_bal = smote_enn.fit_resample(predictors, outcome)




X_train, X_test, y_train, y_test = train_test_split(
    predictors_bal, outcome_bal, test_size=0.3, random_state=1001
    )



from sklearn.ensemble import RandomForestClassifier


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

import pickle
with open('model.pkl','wb') as model_file:
  pickle.dump(clf,model_file)

