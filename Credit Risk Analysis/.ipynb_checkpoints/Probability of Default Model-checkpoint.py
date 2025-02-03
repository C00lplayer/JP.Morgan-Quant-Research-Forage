import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


data = pd.read_csv('Task 3 and 4_Loan_Data.csv')

# Filter The IV and DV depending on what is needed for the model
X = data.drop(columns=["customer_id", "default"])
Y= data["default"]  

# Split the IV and DV for testing and training the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=2, stratify=Y)


# Scale the IV/X values so there isn't a big impact of large scale numbers such as income
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = scaler.fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)


# Create the logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

# Compare the predicted outputs from the model to the actual outputs and display as confusion matrix so analyse accuracy
"""
y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(Y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
disp.plot(cmap="Reds")
plt.show()
"""

# A function which takes in the properties for the loan and predicts the probability of defaulting from the model created and the expected loss
def predict_expected_loss(credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score):
    # Calculating the PD (probability of default)
    new_borrower = pd.DataFrame([[credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]], columns=X.columns) 
    new_borrower_scaled = scaler.transform(new_borrower)
    new_borrower_scaled= pd.DataFrame(new_borrower_scaled, columns= X.columns)
    pd_value = model.predict_proba(new_borrower_scaled)[:, 1][0]
    # Calculating the expected loss of the loan using the formula expected loss= PD * (1-recovery rate) * loan amount
    recovery_rate = 0.10
    expected_loss = pd_value * loan_amt_outstanding * (1 - recovery_rate)
    
    return pd_value, expected_loss

credit_lines_outstanding = int(input("Enter the credit lines outstanding:"))
loan_amt_outstanding= float(input("Enter the loan amount outstanding:"))
total_debt_outstanding = float(input("Enter the total debt outstanding:"))
income = float(input("Enter the income:"))
years_employed= int(input("Enter the years employed:"))
fico_score = int(input("Enter the fico score:"))
pd_value, expected_loss = predict_expected_loss(credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed
                                                , fico_score)

print(f"Probability of Default: {pd_value:.4f}")
print(f"Expected Loss: ${expected_loss:.2f}")