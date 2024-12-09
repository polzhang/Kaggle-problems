import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

imputer = KNNImputer(n_neighbors=5)
scaler = StandardScaler()

df2 = pd.read_csv('test.csv') #test set
df2[['Age', 'Fare']] = imputer.fit_transform(df2[['Age', 'Fare']]) #imputation
df2 = pd.get_dummies(df2, columns=['Sex', 'Embarked']) #one hot encoding for categorical nominal features
df2[['Age','Fare']] = scaler.fit_transform(df2[['Age','Fare']]) #standardising
x_test = df2.drop('PassengerId',axis=1) 


df = pd.read_csv('train.csv') #train set
df[['Age', 'Fare']] = imputer.fit_transform(df[['Age', 'Fare']]) #imputation
df = pd.get_dummies(df, columns=['Sex', 'Embarked']) #one hot encoding for categorical nominal features
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])  #standardising
x_train = df.drop(['Survived','PassengerId'],axis=1)
y_train = df['Survived']


LR = LogisticRegression(max_iter=2000)
LR.fit(x_train,y_train)
y_pred = LR.predict(x_test)
scores = cross_val_score(LR, x_train,y_train, cv=50 , scoring='accuracy') 
print("logistic regression avg accuracy:", scores.mean())

XGB = XGBClassifier(random_state=42, n_estimators=100, learning_rate = 0.1, max_depth = 5)
XGB.fit(x_train, y_train)
y_pred = XGB.predict(x_test)
scores = cross_val_score(XGB, x_train,y_train, cv=5 , scoring='accuracy') 
print("XGBClassifier avg accuracy:", scores.mean())

LGBM = LGBMClassifier(random_state=42, num_leaves = 30, max_depth = 5, learning_rate = 0.05)
LGBM.fit(x_train, y_train)
y_pred = LGBM.predict(x_test)
scores = cross_val_score(LGBM, x_train,y_train, cv=5 , scoring='accuracy') 
print("LGBM avg accuracy:", scores.mean())

y_pred = pd.DataFrame(LGBM.predict(x_test))
y_pred.columns = ['Survived']
results = df2.pop("PassengerId")
results = pd.concat([results, y_pred], axis=1)
results.to_csv('titanicpredictions.csv',index=False)


