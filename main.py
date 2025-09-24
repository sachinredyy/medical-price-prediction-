import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

# Load data
data_df = pd.read_csv("/Users/rahulkumar/Downloads/medical_data.csv")

# See sample data
print(data_df.head())
print(data_df.info())
print(data_df.isnull().sum())
print(data_df.describe())
print(data_df.columns)

# Select numerical columns
df_num = data_df[['age', 'bmi', 'charges']]

# Select categorical columns (excluding 'disease' as it's not in the dataset)
df_cat = data_df[['sex', 'smoker', 'region']]

# Convert categorical variables into dummy/indicator variables
df1 = pd.get_dummies(df_cat, drop_first=True)
print(df1.head())

# Correlation matrix
print(df1.corr())
sns.heatmap(df1.corr(), cmap='RdBu')
plt.show()

# Charges histogram
count, bin_edges = np.histogram(data_df['charges'])
data_df['charges'].plot(kind='hist', xticks=bin_edges, figsize=(20, 12))
plt.title("Patient Charges")
plt.show()

# Import libraries for model training
from sklearn.linear_model import LinearRegression

# Split data into features and target variable
X = pd.concat([df_num.drop(columns='charges'), df1], axis=1)
y = df_num['charges']

# Train-test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Print the options for diseases
print('''Please enter the disease: 
0-Ischemic heart disease, or coronary artery disease
1-Brain Stroke 
2-Lower respiratory infections 
3-Chronic obstructive pulmonary disease
4-Trachea, bronchus, and lung cancers 
5-Diabetes mellitus 
6-Alzheimer’s disease and other dementias 
7-Dehydration due to diarrheal diseases
8-Tuberculosis 
9-Cirrhosis
''')

# Collect user input
age = int(input('Please enter your Age: '))
sex = input('Please enter your gender(m/f): ')
bmi = float(input('Please enter your Body Mass Index: '))

disease = int(input('Select the disease appropriately: '))
smoker = int(input('Do you smoke? \nYes- 1\nNo -0\nPlease enter your answer: '))
region = input('Please select the region where you want to get medicated: \n'
               '0-Hyderabad\n1-Bangalore\n2-Chennai\n3-Delhi\n'
               'Please enter an appropriate option: ')

# Construct input for prediction
x_new_predict = {
    "age": age,
    "bmi": bmi,
    "sex_male": 0 if sex == 'f' else 1,
    "sex_female": 0 if sex == 'm' else 1,
    "smoker_no": 1 if smoker != 1 else 0,
    "smoker_yes": 1 if smoker == 1 else 0,
    "region_Hyderabad": 1 if region == '0' else 0,
    "region_Bangalore": 1 if region == '1' else 0,
    "region_Chennai": 1 if region == '2' else 0,
    "region_Delhi": 1 if region == '3' else 0,
}
# Assuming x_new_predict is a dictionary or similar structure
x_new_predict_values = list(x_new_predict.values())

# Ensure only 7 features are passed
if len(x_new_predict_values) > 7:
    x_new_predict_values = x_new_predict_values[:7]  # Adjust this based on your feature selection

# Make the prediction
y_new_predict = abs(lr.predict([x_new_predict_values]))


# Predicting the cost based on new inputs
#y_new_predict = abs(lr.predict([list(x_new_predict.values())]))

# Display the prediction
print('The average cost of medication is ', y_new_predict[0] * 100)
print('The estimated insurance that can be claimed is ', y_new_predict[0] * 70)
