# %% [markdown]
# Lung cancer, also known as lung carcinoma, is a malignant lung tumor characterized by uncontrolled cell growth in tissues of the lung. This growth can spread beyond the lung by the process of metastasis into nearby tissue or other parts of the body.

# %% [markdown]
# In this study, we tried to predict Lung Cancer  using 6 different algorithm:
# 1. Logistic regression classification
# 2. SVM (Support Vector Machine) classification
# 3. Naive bayes classification
# 4. Decision tree classification
# 5. Random forest classification
# 6. K-Nearest Neighbor classification
# 
# Predictor variable use in classifying lung cancer:
# 1. Age      
# 2. Smokes  
# 3. AreaQ    
# 4. Alkhol
# 
# 
# 

# %% [markdown]
# Import Library

# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix,accuracy_score




# %% [markdown]
# ## Data
# 
# 

# %%
data = pd.read_csv('lung_cancer_examples.csv')


# %%
data.head()

# %% [markdown]
# ## Data Understanding

# %%
data.shape


# %%
data.columns

# %%
data.info()


# %%
data["Result"].nunique()

# %% [markdown]
# As there are two distinct values means we have to do binary classification 

# %%

data["Result"].value_counts().plot(kind="bar")
plt.show()

# %% [markdown]
# ### Profiling Report

# %%
# pandas profiling 
profile = ProfileReport(data,title="Lung Cancer Prediction")
profile.to_file("Lung Cancer Prediction.html")

# %%
data.columns

# %% [markdown]
# ## VISUALIZING THE DATA
# 

# %%
sns.lineplot(x="Age",y="Alkhol",data=data)


# %%
sns.lineplot(x=data['Age'],y=data['Smokes'],data=data)

# %%
data.corr()

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True)
plt.title('Correlation Heatmap of Columns')
plt.show()

# %% [markdown]
# ## Data Preprocessing

# %%
data.isnull().sum().sum()

# %% [markdown]
# - No missing values 

# %%
data1.info()

# %% [markdown]
# <!-- - As most of the columns are numeric and int type, so no need to perform scaling -->

# %% [markdown]
# Eliminate irrelevant variables in analysis such as name, surname

# %%
data1 = data.drop(columns=['Name','Surname'],axis=1)
print(data1.shape)

# %%
data1.head()

# %% [markdown]
# ### Data for training and testing
# 
# 

# %%
from sklearn.model_selection import train_test_split
Y = data1['Result']
X = data1.drop(columns=['Result'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=9)

# %%
print('X train shape: ', X_train.shape)
print('Y train shape: ', Y_train.shape)
print('X test shape: ', X_test.shape)
print('Y test shape: ', Y_test.shape)

# %% [markdown]
# ## Spot Checking

# %%
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))


# %%
results = []

names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# %% [markdown]
# ## Logistic regression classification
# 
# Logistic regression is a technique that can be applied to binary classification problems. This technique uses the logistic function or sigmoid function, which is an S-shaped curve that can assume any real value number and assign it to a value between 0 and 1, but never exactly in those limits. Thus, logistic regression models the probability of the default class (the probability that an input $(X)$ belongs to the default class $(Y=1)$) $(P(X)=P(Y=1|X))$. In order to make the prediction of the probability, the logistic function is used, which allows us to obtain the log-odds or the probit. Thus, the model is a linear combination of the inputs, but that this linear combination relates to the log-odds of the default class.
# 
# Started from make an instance of the model setting the default values. Specify the inverse of the regularization strength in 10. Trained the logistic regression model with the training data, and then applied such model to the test data.

# %%
from sklearn.linear_model import LogisticRegression

# We defining the model
logreg = LogisticRegression(C=10)

# We train the model
logreg.fit(X_train, Y_train)

# We predict target values
Y_predict1 = logreg.predict(X_test)

# accuracy_score = accuracy_score(Y_test, Y_predict1)



# %%
# print(accuracy_score)

# %%

confusion_matrix(Y_test, Y_predict1)

sns.heatmap(confusion_matrix(Y_test, Y_predict1), annot=True)
plt.title('Logistic Regression Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()


# %% [markdown]
# ## Naive bayes classification
# 
# The naive Bayesian classifier is a probabilistic classifier based on Bayes' theorem with strong independence assumptions between the features. Thus, using Bayes theorem $\left(P(X|Y)=\frac{P(Y|X)P(X)}{P(Y)}\right)$, we can find the probability of $X$ happening, given that $Y$ has occurred. Here, $Y$ is the evidence and $X$ is the hypothesis. The assumption made here is that the presence of one particular feature does not affect the other (the predictors/features are independent). Hence it is called naive. In this case we will assume that we assume the values are sampled from a Gaussian distribution and therefore we consider a Gaussian Naive Bayes.

# %%
from sklearn.naive_bayes import GaussianNB

# We define the model
nbcla = GaussianNB()

# We train model
nbcla.fit(X_train, Y_train)

# We predict target values
Y_predict2 = nbcla.predict(X_test)

# %%
# The confusion matrix
confusion_matrix(Y_test, Y_predict2)

sns.heatmap(confusion_matrix(Y_test, Y_predict2), annot=True)
plt.title('Naive Bayes Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()

# %% [markdown]
# ## Decision tree classification
# 
# A decision tree is a flowchart-like tree structure where an internal node represents feature, the branch represents a decision rule, and each leaf node represents the outcome. The decision tree analyzes a set of data to construct a set of rules or questions, which are used to predict a class, i.e., the goal of decision tree is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. In this sense the decision tree selects the best attribute using to divide the records, converting that attribute into a decision node and dividing the data set into smaller subsets, to finally start the construction of the tree repeating this process recursively. 

# %%
from sklearn.tree import DecisionTreeClassifier

# We define the model
dt = DecisionTreeClassifier(random_state=3)

# We train model
dt.fit(X_train, Y_train)

# We predict target values
Y_predict3 = dt.predict(X_test)

# %%
# The confusion matrix
confusion_matrix(Y_test, Y_predict3)

sns.heatmap(confusion_matrix(Y_test, Y_predict3), annot=True)
plt.title('Decision Tree Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()

# %%
Y_test.shape

# %% [markdown]
# ## K-Nearest Neighbor classification
# 
# K-Nearest neighbors is a technique that stores all available cases and **classifies new cases based on a similarity measure (e.g., distance functions)**. This technique is non-parametric since there are no assumptions for the distribution of underlying data and it is lazy since it does not need any training data point model generation. All the training data used in the test phase. **This makes the training faster and the test phase slower and more costlier. In this technique, the number of neighbors k is usually an odd number if the number of classes is 2**. For finding closest similar points,  find the distance between points using distance measures such as Euclidean distance, Hamming distance, Manhattan distance and Minkowski distance.
# 
# 

# %%
from sklearn.neighbors import KNeighborsClassifier

# We define the model
knn = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)

# We train model
knn.fit(X_train, Y_train)

# We predict target values
Y_predict4 = knn.predict(X_test)

# %%
# The confusion matrix


sns.heatmap(confusion_matrix(Y_test, Y_predict4), annot = True)
plt.title('KNN Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()

# %% [markdown]
# ### Test score

# %%
print(accuracy_score(Y_test, Y_predict1))
print(accuracy_score(Y_test, Y_predict2))
print(accuracy_score(Y_test, Y_predict3))
print(accuracy_score(Y_test, Y_predict4))

# %%
# Accuracy scores
accuracy_scores = [accuracy_score(Y_test, Y_predict1),
                   accuracy_score(Y_test, Y_predict2),
                   accuracy_score(Y_test, Y_predict3),
                   accuracy_score(Y_test, Y_predict4)]

# Model names or labels
model_names = ['Logistic Regression', 'Decision Tree', 'Naive Bias', 'KNN']

# Create bar plot
plt.bar(model_names, accuracy_scores, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores of Different Models')
plt.show()


# %% [markdown]
# # Making a Prediction 

# %%
X_test

# %%
age , smokes , areaq , alkhol = 18, 9, 4, 4
prediction1 = logreg.predict([[age , smokes , areaq , alkhol]])
prediction1

# %%
age , smokes , areaq , alkhol = 50, 25, 10, 8
prediction1 = logreg.predict([[age , smokes , areaq , alkhol]])
prediction1

# %%


# %%


# %%


# %%


# %%



