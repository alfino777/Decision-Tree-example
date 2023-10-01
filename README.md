# Decision-Tree-CART5.0-example

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing 

# import some data to play with
iris = pd.read_csv('iris (1).csv')   

iris 


#Complete Iris dataset
label_encoder = preprocessing.LabelEncoder()
iris['Species']= label_encoder.fit_transform(iris['Species']) 


x=iris.iloc[:,0:4]
y=iris['Species']

x 
pd.set_option("display.max_rows", None) 

x 

y


iris['Species'].unique() 

iris.Species.value_counts() 

colnames = list(iris.columns)
colnames

# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)

model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train) 


#PLot the decision tree
tree.plot_tree(model);

fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True); 


model.feature_importances_ 

import pandas as pd
feature_imp = pd.Series(model.feature_importances_,index=fn).sort_values(ascending=False) 
feature_imp

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


preds 

pd.crosstab(y_test,preds)  # getting the 2 way table to understand the correct and wrong predictions


# Accuracy 
np.mean(preds==y_test)

from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


model_gini.fit(x_train, y_train) 

#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test) 

model.feature_importances_ 

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

array = iris.values
X = array[:,0:3]
y = array[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


model = DecisionTreeRegressor()
model.fit(X_train, y_train) 

#Find the accuracy
model.score(X_test,y_test)







               


