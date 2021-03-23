# IMPORTING THE LIBRARIES.
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot
import seaborn as sns

# laoding the dataset as a pandas dataframe.
Data = pd.read_csv("school_grades_dataset.csv")
# there is a strong correlation between G3 (the vector we want to predict) and 
# both G1 and G2 so we will either give them a really small weight or we drop
# both of them. We will drop both of them.
Data = Data.drop('G1', axis = 'columns')
Data = Data.drop('G2', axis = 'columns')

X = Data.iloc[:, :-1].values
Y = Data.iloc[:, 30].values


###################################### DATA PREPROCESSING #############################
# handling the messing values if there are any.

# Xdf = pd.DataFrame(X) # converting X and Y from ndarray to dataframes back again.
# Ydf = pd.DataFrame(Y)
# # checking for messing values 

# print(type(Xdf))
# print(Xdf.isnull().count()) # outputs zero 
# print(Ydf.isnull().count()) # outputs zero 

# # obviously we don't need to handle any messing values because there
# # are zero.

# INITIAL DATA VISUALIZATION.
# how many students got a certain grade.
s = sns.countplot(Data['G3'])
s.set_xlabel("grade", fontsize=15)
s.set_ylabel("number of students", fontsize=15)
plt.show()
# normally distributed 

# we can do visualizaiton as much as we want by comparing every set of variables with each others
# for example we might want to know if having a romantic relationship harms being superior at school or not?

b = sns.swarmplot(x=Data['romantic'], y=Data['G3'])
b.set_xlabel('Romantic Relationship', fontsize = 15)
b.set_ylabel('Grade', fontsize = 15)
plt.show()
# so, relationships does affects grades. 

# we can visualize more features against grades or against each other to capture the 
# correlations between them themselves and between the grades.

# ENCODING CATEGORICAL FEATURES 


labelencoder_X = LabelEncoder()
catfearues = [0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22] # features indicies that needs labelencoding.
# numfeatures =  [2,6,7,12,13,14,23,24,25,26,27,28,29,30,31]
for i in catfearues:
    X[:, i] = labelencoder_X.fit_transform(X[:, i])

# actually we will use onehotencoder here because there is no logical ordering (all the attributes are nominal not ordinal)
# between the categories. we would use ordinalencoder in the state of ordinal attributes.

# onehotendoer = OneHotEncoder()
# X = onehotendoer.fit_transform(X).toarray()

ct = ColumnTransformer(
    [('oh_enc', OneHotEncoder(sparse=False), catfearues)],  # the column numbers I want to apply this to
    remainder='passthrough'  # This leaves the rest of my columns in place
)

X = ct.fit_transform(X) # Notice the output is a string


# SPLITTING THE DATA INTO TRAIN AND TEST SETS.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0, shuffle = True)

# SCALING THE DATASET.
standardscaler = StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_test = standardscaler.transform(X_test)


##################################### TRYING OUT SOME DIFFERENT REGRESSION MODELS #########################
# importing all the regression models i will use.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

# building up the models and append them to models list.
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LR', LinearRegression()))
models.append(('SVR', SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')))
models.append(('RFR', RandomForestRegressor(n_estimators=100)))
models.append(('ETR', ExtraTreesRegressor(n_estimators=100)))

# trying a specific model prediction before looping over all the models.
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, Y_train)
predictions = regressor.predict(X_test)

# error measures for evaluation
RMSE = [] # means suqared error
MAE = [] # mean absolute error

# this loop will fit the regressor to the data and it will also calculate 
# both the MAE and the RMSE errors.
for name, regressor in models:
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    mse = mean_squared_error(Y_test, predictions)
    RMSE.append(sqrt(mse))
    MAE.append(mean_absolute_error(Y_test, predictions))

# printing out both the MSE and MAE errors.
print(f"Mean Square error: {RMSE}\n\nMean absolute error: {MAE} \n\n ")

 



