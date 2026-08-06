import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the data from the Excel file
file_path = r"D:\4 TH YEAR\FINAL YEAR PRO\GrapesDataset\grape_features.xlsx"
data = pd.read_excel(file_path)

# Step 2: Display the first few rows of the data to understand its structure
print("Initial Data:")
print(data.head())

# Step 3: Data Cleaning
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill or drop missing values if necessary
data.fillna(0, inplace=True)  # Replace NaN with 0, or you can use data.dropna()

# Step 4: Define features (X) and labels (y)
# Assuming the features are all columns except 'label'
X = data.drop(columns=['label'])
y = data['label']  # This should be the label column you created earlier

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Normalize the features (optional but recommended for many ML models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Create a DataFrame for scaled features for verification (optional)
df_train = pd.DataFrame(X_train_scaled, columns=X.columns)
df_train['label'] = y_train.reset_index(drop=True)

df_test = pd.DataFrame(X_test_scaled, columns=X.columns)
df_test['label'] = y_test.reset_index(drop=True)

# Output the prepared data for review
print("\nTraining Features:")
print(df_train.head())
print("\nTesting Features:")
print(df_test.head())

# Optional: Save the prepared data to new Excel files
train_output_path = r"D:\4 TH YEAR\FINAL YEAR PRO\GrapesDataset\grape_features_train.xlsx"
test_output_path = r"D:\4 TH YEAR\FINAL YEAR PRO\GrapesDataset\grape_features_test.xlsx"

df_train.to_excel(train_output_path, index=False)
df_test.to_excel(test_output_path, index=False)

print(f"\nPrepared training data saved to {train_output_path}")
print(f"Prepared testing data saved to {test_output_path}")
