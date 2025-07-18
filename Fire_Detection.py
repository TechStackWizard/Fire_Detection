import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import  accuracy_score,classification_report,ConfusionMatrixDisplay
# from sklearn.neighbors import KNeighborsClassifier
# # from xgboost import XGBClassifier
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df1 = pd.read_csv('modis_2021_India.csv')
df2 = pd.read_csv('modis_2022_India.csv')
df3 = pd.read_csv('modis_2023_India.csv')

# print("Data loaded succefully..")
# print(df1.head())

df = pd.concat([df1, df2, df3], ignore_index=True)
# print(df.head(10))

# print(df.shape) #it prints rows and cols

# print(df.info()) #It gives information about Columns DataType

# print(df.isnull().sum()) #Prints the missing values

# print(df.duplicated().sum()) #Count of duplicate values

# print(df.columns) #All the columns(features) name present in Dataset

# print(df.describe().T)

# print(df.type.value_counts()) #checks Unique values of target variable

# Check unique and n unique for all categorical features

# for col in df.columns:
#     if df[col].dtype == 'object':
#         print(f"Column : {col}")
#         print(f"Unique values : {df[col].unique()}")
#         print(f"Number of unique values : {df[col].nunique()}")
#         print("-" * 50)



#Count plot for type

# plt.figure(figsize=(8,6)) #screen size
# sns.countplot(x='type', data=df)
# plt.title("Distribution of Fire Types")
# plt.xlabel('Fire Type')
# plt.ylabel('Count')
# plt.show()

#Historical of confidence

plt.figure(figsize=(8,6))
sns.histplot(df['confidence'], bins=20, kde=True)
plt.title("Distribution of confidence")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.show()








