import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples (rows) in the dataset
n_samples = 1000

# Generate 28 random features (V1 to V28) from a normal distribution
V_columns = [f'V{i}' for i in range(1, 29)]  # Feature names from V1 to V28
X = np.random.randn(n_samples, 28)  # 28 features with random values

# Generate the 'Amount' column (random transaction amounts between 1 and 1000)
Amount = np.random.uniform(1, 1000, size=n_samples)

# Generate the 'Time' column (random times in seconds since the start of the day)
Time = np.random.randint(1, 86400, size=n_samples)  # 1 day = 86400 seconds

# Generate the 'Class' column (0 for non-fraud, 1 for fraud)
# Set 1% of the samples to be fraud (Class = 1), and the rest to be non-fraud (Class = 0)
Class = np.concatenate([np.zeros(n_samples - 30), np.ones(30)])  # 30 fraud cases out of 1000
np.random.shuffle(Class)  # Shuffle the fraud labels to distribute them randomly

# Combine all the features into a DataFrame
data = pd.DataFrame(X, columns=V_columns)
data['Amount'] = Amount
data['Time'] = Time
data['Class'] = Class

# Save the dataset to a CSV file
data.to_csv('creditcard.csv', index=False)

# Show the first few rows of the generated dataset
print("Dataset successfully created!")
print(data.head())