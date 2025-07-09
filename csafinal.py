import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
#importing the dataset and reading it
diabetes = pd.read_csv('diabetes.csv')
#displaying the dataset
#grouping the dataset by the outcome where 0 is no diabetes and 1 is diabetes
diabetes.groupby('Outcome').mean()
#in ML X is basically what causes the outcome so we drop the outcome column and assign it to X
X = diabetes.drop(columns = 'Outcome', axis = 1)
#while Y is the effects so we assign Y only the outcome column
y = diabetes['Outcome']
#scaling the inconsistent data to fit a understanable range
scaler= StandardScaler()
new_data = scaler.fit_transform(X)
#assigning the scaled data to X
X = new_data
#splitting the data into training and testing where 40% is testing and 60% is training and random state is 1, random state is used to make the data split consistent and is never random, like a minecraft seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1)
#stratifiedKFold is used to split the data into 3 folds and shuffle the data
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#scaling the data
X_train_scaled_features = scaler.fit_transform(X_train)
X_test_scaled_features=scaler.transform(X_test)
#using the Random Forest Classifier because it seemed to have the best result; Standardizing X made it go down from 80% to 79.5%, but it seemed to be the only way it would work
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_scaled_features, y_train)
#predicting the test data
y_pred= rfc.predict(X_test_scaled_features)
#displaying the accuracy of the classifier
rfc_a=accuracy_score(y_pred=y_pred,y_true=y_test)
#displaying the f1 score of the classifier
rfc_f1=f1_score(y_pred=y_pred,y_true=y_test,average='weighted')
#displaying the accuracy and f1 score of the classifier
print(f'The accuracy of the Random Forest Classifier is {rfc_a}')
print(f'The f1 score of the Random Forest Classifier is {rfc_f1}')
#displaying the actual and predicted values
y_pred = rfc.predict(X_test_scaled_features)
st.title("       The Predicition models accuracy")
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#plotting the actual and predicted values
fig, ax = plt.subplots()
# Plot the actual outcome values as a line graph
ax.plot(result_df.index, result_df['Actual'], label='Actual', linewidth=2)
# Plot the predicted outcome values as a line graph
ax.plot(result_df.index, result_df['Predicted'], label='Predicted', linewidth=2)
# Label the x-axis as 'Data Point'
ax.set_xlabel('Data Point')
# Label the y-axis as 'Outcome where the only options are 1 or 0'
ax.set_ylabel('Outcome (1 or 0)')
# Set the title of the plot to actual vs predicted values
ax.set_title('Actual vs Predicted Values - Line Graph')
# Display the legend to differentiate between actual and predicted
ax.legend()
# Show the plot of the actual and predicted values
st.pyplot(fig)
#cross-validation scores of the classifier
scores = cross_val_score(rfc, X, y, cv=skf)
#displaying the cross-validation scores of the classifier
print("Cross-validation scores:", scores)
#displaying the mean score of the classifier
print("Mean score:", scores.mean())








st.title("Diabetes Risk Prediction")
st.write("Enter your data to predict your risk of diabetes")

col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=0)
    Glucose = st.number_input("Glucose", min_value=0, max_value=200, value=0)
    BloodPressure = st.number_input("BloodPressure", min_value=0, max_value=122, value=0)
    SkinThickness = st.number_input("SkinThickness", min_value=0, max_value=99, value=0)

with col2:
    Insulin = st.number_input("Insulin", min_value=0, max_value=846, value=0)
    BMI = st.number_input("BMI", min_value=0, max_value=100, value=0)  
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0, max_value=4, value=0)
    Age = st.number_input("Age", min_value=0, max_value=81, value=0)

# Add submit button in the center
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submit_button = st.button("Predict Diabetes Risk", use_container_width=True)

# Only show results if button is clicked
if submit_button:
    input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

    #Turns input data into an array
    input_data = np.array(input_data).reshape(1, -1)
    #Standardizes the data in the array
    standardized_data = scaler.transform(input_data)

    # Ensure that the shape matches training data
    st.write("Standardized Input Data:", standardized_data)

    #Prediction from
    prediction = rfc.predict(standardized_data)
    #final output
    if prediction[0] == 0:
        st.write("You probably do not have diabetes.")
    else:
        st.write("You should go get checked by the doctor for diabetes.")
