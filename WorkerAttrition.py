import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate
from xgboost import XGBClassifier

#Load data from file
file_path = "C:/Users/Neel/Desktop/2023-2024/Spring 2024/Big Data and Forecasting/assignment4_HRemployee_attrition.csv"
data = pd.read_csv(file_path)
#print(data1)

# Convert some of the variables into binary values
binary_mapping = {'Attrition': {'Yes': 1, 'No': 0}, 'Gender': {'Male': 1, 'Female': 0},
   'Over18': {'Y': 1, 'N': 0}, 'OverTime': {'Yes': 1, 'No': 0}}


# Apply binary values to each relevant column
for column, mapping in binary_mapping.items():
   data[column] = data[column].map(mapping)
   
#Make other variables numerical values
#Use this to classify which columns you want to assign numerical values to (exclude columns mentioned above)
categorical_columns = [col for col in data.columns if data[col].dtype == 'object' and col not in binary_mapping]
# Specify which columns to use
data[['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance']] = data[['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance']].astype(str)

categorical_columns.extend(['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance'])
# Make above columns have numerical values
numerical_var = pd.get_dummies(data[categorical_columns])


# Drop all original values that were used for transformation above and add transformed code into main data set
data.drop(categorical_columns, axis=1, inplace=True)
data = pd.concat([data, numerical_var], axis=1)

# Scale columns (excluding binary columns): Isolate columns that haven't been transformed
continous_columns = [col for col in data.columns if col not in binary_mapping.keys() and col not in numerical_var.columns]

scaler = StandardScaler()
data[continous_columns] = scaler.fit_transform(data[continous_columns])


# Define the dependent variable and variable
y = data['Attrition']
X = data.drop('Attrition', axis=1)


# Neural network

mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='tanh', alpha=0.0001, max_iter=500, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model on the training data
mlp.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test)

# Set up 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation and get the accuracy scores
cv_scores = cross_val_score(mlp, X, y, cv=cv, scoring='accuracy')

# Display the cross-validation scores
print("Cross-validation scores:", cv_scores)
print(f"Mean accuracy: {cv_scores.mean():.4f}")

# XGBClassifier
xgb_classifier = XGBClassifier(random_state=42)

# 10 fold cross-validation for XGBClassifier
xgb_scores = cross_val_score(xgb_classifier, X, y, cv=10)

# Print the mean accuracy and standard deviation of the scores
print("XGBoost Classifier Accuracy:", xgb_scores.mean())

# Train XGB Classifier
xgb_classifier.fit(X, y)

# Get feature importances
feature_importances = xgb_classifier.feature_importances_

# Get feature names
feature_names = X.columns

# Create a DataFrame to display feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances
#print(importance_df)

# Turn importance_df into a table
table = tabulate(importance_df, headers='keys', tablefmt='pretty', showindex=False)

print("Feature Importances:")
print(table)