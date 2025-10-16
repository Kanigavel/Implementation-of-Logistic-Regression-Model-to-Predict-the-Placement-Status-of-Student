# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary Python libraries.

2.Load the dataset containing student details and placement status.

3.Perform exploratory data analysis (EDA) and handle missing values if any.

4.Split the dataset into training and testing data.

5.Use Logistic Regression to train the model.

6.Predict placement status for the test data.

7.Evaluate the model using accuracy and confusion matrix.

8.Visualize the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Kanigavel M
RegisterNumber: 212224240070 
*/
# Implementation of Logistic Regression Model to Predict the Placement Status of Students

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Step 2: Load the dataset
data = pd.read_csv("Placement_Data.csv")  # Ensure your CSV file is in the same directory
print("First five records:")
print(data.head())

# Step 3: Preprocessing
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['ssc_b'] = le.fit_transform(data['ssc_b'])
data['hsc_b'] = le.fit_transform(data['hsc_b'])
data['hsc_s'] = le.fit_transform(data['hsc_s'])
data['degree_t'] = le.fit_transform(data['degree_t'])
data['workex'] = le.fit_transform(data['workex'])
data['specialisation'] = le.fit_transform(data['specialisation'])
data['status'] = le.fit_transform(data['status'])  # Placed=1, Not Placed=0

# Step 4: Define independent (X) and dependent (Y) variables
X = data[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']]
Y = data['status']

# Step 5: Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 6: Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Step 7: Make predictions
Y_pred = model.predict(X_test)

# Step 8: Combine actual and predicted values in a table
output_table = pd.DataFrame({
    'SSC %': X_test['ssc_p'].values,
    'HSC %': X_test['hsc_p'].values,
    'Degree %': X_test['degree_p'].values,
    'E-Test %': X_test['etest_p'].values,
    'MBA %': X_test['mba_p'].values,
    'Actual Status': Y_test.values,
    'Predicted Status': Y_pred
})

# Step 9: Display the results
print("\n--- Actual vs Predicted Placement Status ---\n")
print(output_table.head(10))  # Display first 10 rows

# Step 10: Evaluate the model
print("\nModel Evaluation Metrics:")
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

# Step 11: Visualization
plt.scatter(X_test['mba_p'], Y_test, color='red', label='Actual')
plt.scatter(X_test['mba_p'], Y_pred, color='blue', label='Predicted')
plt.title("Actual vs Predicted Placement Status")
plt.xlabel("MBA Percentage")
plt.ylabel("Placement Status (0=Not Placed, 1=Placed)")
plt.legend()
plt.show()

```

## Output:
<img width="1123" height="675" alt="Screenshot 2025-10-16 133044" src="https://github.com/user-attachments/assets/807c74f5-b363-4f09-b46e-3ec79a94b240" />
<img width="1080" height="871" alt="image" src="https://github.com/user-attachments/assets/aff91de6-9591-4398-99d7-44b97f74402d" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
