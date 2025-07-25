import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('student_data.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Separate features and target
X = df.drop(['student_id', 'status'], axis=1)
y = df['status']

# Convert target to binary (0: fail, 1: pass)
y = y.map({'fail': 0, 'pass': 1})

# Store feature names and categorical columns for later use
feature_names = X.columns.tolist()
categorical_columns = ['gender', 'parent_education_level', 'parent_income', 'school_type',
                      'extra_classes', 'test_preparation_course', 'peer_group_study',
                      'library_usage', 'sports_participation', 'mental_stress_level',
                      'learning_disability', 'family_support']

# Create label encoders for categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
numerical_columns = [col for col in X.columns if col not in categorical_columns]
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save the model and preprocessing objects
model_data = {
    'model': model,
    'label_encoders': label_encoders,
    'scaler': scaler,
    'feature_names': feature_names,
    'categorical_columns': categorical_columns,
    'numerical_columns': numerical_columns
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nModel and preprocessing objects saved to 'model.pkl'")

# Save feature importance for reference
feature_importance.to_csv('feature_importance.csv', index=False)
print("Feature importance saved to 'feature_importance.csv'")