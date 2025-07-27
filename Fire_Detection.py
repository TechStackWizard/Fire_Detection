import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  accuracy_score,classification_report,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE

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

# plt.figure(figsize=(8,6))
# sns.histplot(df['confidence'], bins=20, kde=True)
# plt.title("Distribution of confidence")
# plt.xlabel("Confidence")
# plt.ylabel("Frequency")
# plt.show()





# Week 2 ML practices for Deforestation Detection

# Box plot for 'confidence' by 'type'
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='type', y='confidence', data=df)
# plt.title('Confidence by Fire Type')
# plt.xlabel('Fire Type')
# plt.ylabel('Confidence')
# plt.show()


# Scatter plot of 'latitude' vs 'longitude'
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x='longitude', y='latitude', data=df, hue='type', s=10)
# plt.title('Fire Locations by Type')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend(title='Fire Type')
# plt.show()

# Count plot for 'daynight'
# plt.figure(figsize=(6, 4))
# sns.countplot(x='daynight', data=df)
# plt.title('Distribution of Day/Night Observations')
# plt.xlabel('Day/Night')
# plt.ylabel('Count')
# plt.show()

# Count plot for 'Satellite'
# plt.figure(figsize=(6, 4))
# sns.countplot(x='satellite', data=df)
# plt.title('Distribution of Satellite Observations')
# plt.xlabel('Satellite')
# plt.ylabel('Count')
# plt.show()

# # Count plot for 'version'
# plt.figure(figsize=(6, 4))
# sns.countplot(x='version', data=df)
# plt.title('Distribution of Version')
# plt.xlabel('Version')
# plt.ylabel('Count')
# plt.show()

#this code take more time
#Pairplot for numerical features (subset)
# sns.pairplot(df[['latitude', 'longitude', 'brightness', 'confidence', 'frp', 'type']], hue='type', diag_kind='kde')
# plt.suptitle('Pairplot of Numerical Features')
# plt.show()


# Heatmap of correlations between numerical features
# plt.figure(figsize=(10, 8))
# correlation_matrix = df[['latitude', 'longitude', 'brightness', 'confidence', 'frp']].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap of Numerical Features')
# plt.show()

# numerical_cols = df.select_dtypes(include=np.number).columns
# print(numerical_cols)


numerical_cols = ['brightness', 'scan', 'track', 'acq_time','confidence', 'version', 'bright_t31', 'frp']
# df[numerical_cols].hist(bins=50, figsize=(15, 10))
# plt.suptitle('Histograms of Numerical Features')
# plt.show()


# import statsmodels.api as sm
# import scipy.stats as stats

# # List of numerical features to check for distribution
# numerical_features = ['brightness', 'confidence', 'frp', 'bright_t31', 'scan', 'track']
# numerical_features = ['confidence']

# for feature in numerical_features:
#     print(f"Analyzing distribution for: {feature}")

#     # KDE Plot
#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)
#     sns.kdeplot(df[feature], fill=True)
#     plt.title(f'KDE Plot of {feature}')
#     plt.xlabel(feature)
#     plt.ylabel('Density')

#     # QQ Plot
#     plt.subplot(1, 2, 2)
#     stats.probplot(df[feature], dist="norm", plot=plt)
#     plt.title(f'QQ Plot of {feature}')

#     plt.tight_layout()
#     plt.show()
#     print("-" * 50)



    # --- Temporal Analysis ---
# Convert 'acq_date' to datetime objects
df['acq_date'] = pd.to_datetime(df['acq_date'])
# Extract temporal features
df['year'] = df['acq_date'].dt.year
df['month'] = df['acq_date'].dt.month
df['day_of_week'] = df['acq_date'].dt.dayofweek # Monday=0, Sunday=6
df['day_of_year'] = df['acq_date'].dt.dayofyear
df['hour'] = df['acq_time'].astype(str).str[:2].astype(int) # Assuming acq_time is HHMM


# Visualize fire detections over months
# plt.figure(figsize=(10, 6))
# sns.countplot(data=df, x='month', palette='viridis')
# plt.title('Fire Detections by Month (2023)')
# plt.xlabel('Month')
# plt.ylabel('Number of Detections')
# plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# plt.show()


# Visualize fire detections by day of the week
# plt.figure(figsize=(10, 6))
# sns.countplot(data=df, x='day_of_week', palette='viridis')
# plt.title('Fire Detections by Day of Week (2023)')
# plt.xlabel('Day of Week')
# plt.ylabel('Number of Detections')
# plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
# plt.show()





# Visualize outliers using box plots for key numerical features
# plt.figure(figsize=(12, 8))
# sns.boxplot(data=df[numerical_cols])
# plt.title('Box Plots for Key Numerical Features')
# plt.ylabel('Value')
# plt.show()


def remove_outliers_iqr(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
  return df_cleaned

# Apply outlier removal to numerical columns
for col in numerical_cols:
  df = remove_outliers_iqr(df, col)

# print("Shape after removing outliers:", df.shape)




# Visualize box plots after outlier removal
# plt.figure(figsize=(12, 8))
# sns.boxplot(data=df[numerical_cols])
# plt.title('Box Plots for Numerical Features After Outlier Removal')
# plt.ylabel('Value')
# plt.show()

# print(df.head())

# print(df.type.value_counts())

# categorical_cols = df.select_dtypes(include='object').columns
# print(categorical_cols)

# Select categorical columns for encoding
categorical_cols_to_encode = ['daynight', 'satellite', 'instrument']

# # Apply One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)


# print(df_encoded.head(100))

# print(df_encoded.type.value_counts())

# !pip install folium
# import folium

# # # Create map and sample data
# india_map = folium.Map(location=[22.351115, 78.667743], zoom_start=5)
# sample_df = df_encoded.sample(n=min(10000, len(df_encoded)), random_state=42)

# # Add markers
# for _, row in sample_df.iterrows():
#     folium.CircleMarker(
#         location=[row['latitude'], row['longitude']],
#         radius=3,
#         color='red',
#         fill=True,
#         fill_opacity=0.6,
#         popup=f"FRP: {row['frp']:.2f}, Date: {row['acq_date'].strftime('%Y-%m-%d')}"
#     ).add_to(india_map)

# # Save the map to an HTML file and open it in the default browser
# india_map.save("india_fire_map.html")
# import webbrowser
# webbrowser.open("india_fire_map.html")


# scaler = StandardScaler()
# numerical_cols_to_scale = ['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']
# df_encoded[numerical_cols_to_scale] = scaler.fit_transform(df_encoded[numerical_cols_to_scale])
# print(df_encoded.head())

# print(df_encoded.info())

# Heatmap of correlations between numerical features
plt.figure(figsize=(10, 8))
correlation_matrix = df_encoded[['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
# plt.show()

# print(df_encoded.head())

# print(df_encoded.type.value_counts())

# Separate features (X) and target variable (y)
# Assuming 'type' is the target variable you want to predict
# Drop temporal features if not intended for prediction task that uses 'type' as target
features = ['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']
target = 'type'

X = df_encoded[features]
y = df_encoded[target]

# print(X)
# print(y)


smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the distribution of the target variable after resampling
print("Distribution of target variable after SMOTE:")
print(y_resampled.value_counts())