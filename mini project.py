1.   #DATA COLLECTION#####
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import scipy.stats as norm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

df= pd.read_csv(r'C:/Users/nehad/OneDrive/Desktop/divya_project.csv')
df.head()
df.shape
df.info()
df.describe

2.       #Exploratry DATA ANALYSIS #######
#FIRST MOMENT BUSINESS###

####### MEAN #########
df.mean()
#####MEDIAN##########
df.median()
##MODE##
df.mode()

#SECOND MOMENT BUSINESS#####
df.var()
##std variance###
df.std()
##THIRD MOMENT BUSINESS#(skewness)#
df.skew()
#patient_id show the negative value.means its negative skewness.
###FOURTH MOMENT BUSINESS## (kurtosis)
df.kurt()
#all value are positive.means all are positive kurtosis,,,high peak, high frequency ,and thicker tail.
#########column########
df.columns

##### SPECIFIC COLUMN DESCRIBE###########
df[['Dept' , 'Quantity']].describe()

3.#   GRAPHIC REPRESENTATION    #################
#####HISTOGRAM###
columns_to_plot = ["Patient_ID", "Quantity", "ReturnQuantity",
                   "Final_Cost", "Final_Sales", "RtnMRP"]
for column in columns_to_plot:
    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins=1000, color='blue', alpha=0.7)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
    
    #DENSITY PLOT##
columns_to_plot = ["Patient_ID", "Quantity", "ReturnQuantity",
                   "Final_Cost", "Final_Sales", "RtnMRP"]
sns.set(style="whitegrid")
for column in columns_to_plot:
    plt.figure(figsize=(7, 6))
    sns.kdeplot(data=df, x=column, fill=True, color='blue')
    plt.title(f'Density Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.show()
    ####BOX PLOT###
columns_to_plot = ["Patient_ID", "Quantity", "ReturnQuantity",
                   "Final_Cost", "Final_Sales", "RtnMRP"]
for column in columns_to_plot:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, y=column, color='orange')
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.show()
  ####SCATTER PLOT######
plt.scatter(x=df['Patient_ID'],y=df['Quantity'],color ='RED')
#####PAIR PLOT####
sns.pairplot(df)
4. ###AUTO EDA#####

import sweetviz as sv
s=sv.analyze(df)
s.show_html()

5.     #DATA PREPROCESSING##
 #TYPECASTING### changing datatype column whichever required
df.dtypes
df.Final_Cost=df.Final_Cost.astype("int64")
df.Final_Cost=df.Final_Sales.astype("int64")
df.RtnMRP=df.RtnMRP.astype("int64")
df.dtypes 
#checking for duplicates#
df.duplicated().sum()
#drop duplicates value#
df=df.drop_duplicates()
#checking duplicates removed or not#
df.duplicated().sum()
#correlation Co-efficient
df.corr()

6.   # outliers####

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, orient="v")
plt.title("Box Plot for Dataset Columns")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#Proper Visualization of outliers.
df_num = df.select_dtypes(exclude= ["object"])
df_num.plot(kind = "box" , subplots = True , sharey = False , figsize= (18,8))

##OUTLIER TREATMENT####
sns.boxplot(df.Patient_ID)
sns.boxplot(df.RtnMRP)
sns.boxplot(df.ReturnQuantity)
sns.boxplot(df.Final_Cost)
sns.boxplot(df.Final_Sales)

7.#WINSORIZATION USING IQR#
from feature_engine.outliers import Winsorizer
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                          tail = 'both',
                          fold = 1.5, 
                          variables = ["Patient_ID", "Quantity", 
                                             "Final_Cost", "Final_Sales"])
df_IQR=winsor_iqr.fit_transform(df[["Patient_ID", "Quantity",
                   "Final_Cost", "Final_Sales"]])
sns.boxplot(df_IQR)

#using gaussian##

winsor_gaussian = Winsorizer(capping_method = 'gaussian', 
                          tail = 'both',
                          fold = 3,
                          variables = ['RtnMRP'])

df_t = winsor_gaussian.fit_transform(df[['RtnMRP']])
sns.boxplot(df_t.RtnMRP)

##ZERO AND NEAR ZERO VARIANCE###
df.var()

#BINARIZATION
df_binarization= pd.cut(df['Final_Cost'], 
                              bins = [min(df.Final_Cost), df.Final_Cost.mean(), max(df.Final_Cost)],
                              labels = ["First", "Last"])
df_binarization.value_counts()

# DUMMY VARIABLE CREATION:
df.SubCat.value_counts()
df.SubCat1.unique()
########Q-Q PLOT(FOR CHECKING NORMALIY)##
import pylab
stats.probplot(df_IQR.Final_Cost,dist="norm",plot=pylab)
###its non-normal distribution....
stats.probplot(df_IQR.Patient_ID,dist="norm",plot=pylab)
#.nonnormal distribution#
##use data xmation###(johnson xform)
from feature_engine import transformation

tf = transformation.YeoJohnsonTransformer(variables =[ 'Final_Cost'])
tf = transformation.YeoJohnsonTransformer(variables =[ 'Final_Sales'])
Sales_tf=tf.fit_transform(df_IQR),
Cost_tf = tf.fit_transform(df_IQR)
##boxcox xmation##
transformed_column, lambda_value = stats.boxcox(df_IQR['Final_Cost'])

df_IQR['Final_Cost' + '_BoxCox'] = transformed_column
stats.probplot(transformed_column, dist = stats.norm, plot = pylab)
#STANDARDIZATION AND NORMALIZATION###
df.dtypes
df_scaled=df.copy()
col_names =["Patient_ID", "Quantity", "Final_Cost", "Final_Sales","ReturnQuantity" , "RtnMRP", ]

features =df_scaled[col_names]

minmaxscale = MinMaxScaler()
df_scaled[col_names]= minmaxscale.fit_transform(features.values)
df_scaled
####ENCODER#####
df = df[['Typeofsales', 'Patient_ID', 'Specialisation', 'Dept', 'Dateofbill',
       'Quantity', 'ReturnQuantity', 'Final_Cost', 'Final_Sales', 'RtnMRP',
       'Formulation', 'DrugName', 'SubCat', 'SubCat1']]
enc = OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(df).toarray())
enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:, 2:]).toarray())
enc.fit_transform(df.iloc[:, 2:]).toarray()
####LABEL ENCODER####
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X= df.iloc[:14218, :14]
X['Typeofsales1'] = le.fit_transform(X['Typeofsales'])
X['Specialisation1'] = le.fit_transform(X['Specialisation'])
X['Dept1'] = le.fit_transform(X['Dept'])
X['Dateofbill1'] = le.fit_transform(X['Dateofbill'])
X['Formulation1'] = le.fit_transform(X['Formulation'])
X['DrugName1'] = le.fit_transform(X['DrugName'])
X['SubCat1'] = le.fit_transform(X['SubCat'])
X['SubCat11'] = le.fit_transform(X['SubCat1'])
###DATABASE CONNECTIVITY WITH SQL###
pip install pandas sqlalchemy pymysql
pip install --upgrade ipython jupyter
pip install --upgrade sqlalchemy
from sqlalchemy import create_engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",# user
                               pw = "Jamizanny17", # passwrd
                               db = "divya")) #database
df.to_sql('df', con = engine, if_exists = 'replace', chunksize = None, index= False)
sql = "SELECT * FROM divya_project;"
