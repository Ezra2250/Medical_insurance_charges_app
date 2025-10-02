import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load cleaned insurance dataset
insurance_df = pd.read_csv("insurance_data.csv", index_col=0)

# Split into features (X) and target (y)
X = insurance_df.iloc[:, 0:6]   # first 6 columns as features
y = insurance_df.iloc[:, -1]    # last column as target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "insurance_app.pkl")
print("✅ Model trained and saved successfully as insurance_app.pkl")
