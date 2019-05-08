import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cProfile import label

diabetes = pd.read_csv("C:/Users/ksaldanh/Documents/Diabetes.csv")
print("Dimensions of the diabetes dataset is {}".format(diabetes.shape))

#In the dataset we are interested in the outcome variable, Here 0 means no diabetes and 1 means diabtes,
#so lets counts the number of 0s and 1s 
print(diabetes.groupby("Outcome").size())

#EXPLORATORY DATA ANALYSIS(EDA)
#Lets visualize it because pictures speak a thousand words
import seaborn as sns
sns.countplot(diabetes["Outcome"], label="Count")
plt.show()

#Lets check the dataframe for any missing values
print(diabetes.isnull().sum())
print(diabetes.describe())
#After observing the dataset we can observe that there are 0s as values in Blood Glucose, BloodPressure, skin thickness,
#Insulin, BMI which is most likely not possible
#Deleting the values could result in loss of valuable information in other predictors
#Imputing mean/average values is not the smartest idea in this case as it is individual readings
#So we conclude that the dataset is not reliable

