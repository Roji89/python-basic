from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


##################### 1. Supervised Learning ######################
##################### A. Linear Regression  #######################
# Predicting Tax Refund Amount
# The tax refund amount can be predicted based on income, deductions, and credits.
# Training data: [income, deductions, credits]
X_train_A = [
    [50000, 10000, 1500],
    [60000, 12000, 2000],
    [70000, 15000, 2500],
    [80000, 20000, 3000]
]

# Corresponding refund amounts
y_train_A = [2000, 2500, 3000, 3500]
model = LinearRegression()
model.fit(X_train_A, y_train_A)  # X = [income, deductions, credits], y = refund_amount
predicted_refund = model.predict([[75000, 12000, 2000]])
print(predicted_refund)

################### B. Logistic Regression => binary outcome(yes/no) ###################
# Training data: [income, loan_amount, credit_score]
X_train_B = [
    [50000, 20000, 700],
    [60000, 25000, 750],
    [40000, 15000, 650],
    [80000, 30000, 800],
    [30000, 10000, 600]
]

# Corresponding loan approval status (1 = approved, 0 = not approved)
y_train_B = [1, 1, 0, 1, 0]

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_B, y_train_B)

# Predict loan approval for a new applicant
new_applicant = [[55000, 22000, 720]]  # [income, loan_amount, credit_score]
predicted_approval = model.predict(new_applicant)
print(f"Logistic Regression: {'Approved' if predicted_approval[0] == 1 else 'Not Approved'}")

# Probability of approval
approval_probability = model.predict_proba(new_applicant)
print(f"Logistic Regression: {approval_probability[0][1]:.2f}")

##################### C. Decision Tree  ######################
# Training data: [income, loan_amount, credit_score]
X_train_C = [
    [50000, 20000, 700],
    [60000, 25000, 750],
    [40000, 15000, 650],
    [80000, 30000, 800],
    [30000, 10000, 600]
]

# Corresponding loan approval status (1 = approved, 0 = not approved)
y_train_C = [1, 1, 0, 1, 0]

# Create and train the Decision Tree model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train_C, y_train_C)

# Predict loan approval for a new applicant
new_applicant = [[55000, 22000, 720]]  # [income, loan_amount, credit_score]
predicted_approval = model.predict(new_applicant)
print(f"Decision Tree: {'Approved' if predicted_approval[0] == 1 else 'Not Approved'}")

# Training data: [income, deductions, credits]
X_train_C = [
    [50000, 10000, 1500],
    [60000, 12000, 2000],
    [70000, 15000, 2500],
    [80000, 20000, 3000]
]

# Corresponding refund amounts
y_train_C = [2000, 2500, 3000, 3500]

# Create and train the Decision Tree model
model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(X_train_C, y_train_C)

# Predict refund for a new case
predicted_refund = model.predict([[75000, 12000, 2000]])
print(f"Decision Tree: {predicted_refund[0]:.2f}")

###################### D.Random Forest  ######################
# Random Forest for Classification
# Training data: [income, loan_amount, credit_score]
X_train = [
    [50000, 20000, 700],
    [60000, 25000, 750],
    [40000, 15000, 650],
    [80000, 30000, 800],
    [30000, 10000, 600]
]

# Corresponding loan approval status (1 = approved, 0 = not approved)
y_train = [1, 1, 0, 1, 0]

# Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict loan approval for a new applicant
new_applicant = [[55000, 22000, 720]]  # [income, loan_amount, credit_score]
predicted_approval = model.predict(new_applicant)
print(f"random forest classification: {'Approved' if predicted_approval[0] == 1 else 'Not Approved'}")

# random forest for regression
# Training data: [income, deductions, credits]
X_train = [
    [50000, 10000, 1500],
    [60000, 12000, 2000],
    [70000, 15000, 2500],
    [80000, 20000, 3000]
]

# Corresponding refund amounts
y_train = [2000, 2500, 3000, 3500]

# Create and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict refund for a new case
predicted_refund = model.predict([[75000, 12000, 2000]])
print(f"random forest regression: {predicted_refund[0]:.2f}")

