# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Step 1: Load and Explore the Dataset

# Loading the Iris dataset from sklearn
iris = load_iris()

# Converting the Iris dataset into a pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Adding the target variable (species) to the DataFrame
df['species'] = iris.target

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Step 2: Data Exploration

# Check data types and missing values
print("\nDataset Info (Data types and non-null counts):")
print(df.info())

# Check for any missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Basic statistics of the numerical columns
print("\nDescriptive statistics (mean, std, etc.):")
print(df.describe())

# Group by 'species' and calculate the mean for numerical columns
print("\nGrouped by species (mean values):")
grouped = df.groupby('species').mean()
print(grouped)

# Step 3: Data Visualization

# 1. Line Chart (Simulated time-series data for Sales)
import numpy as np

# Simulating monthly sales data for the year 2024
time = pd.date_range('2024-01-01', periods=12, freq='M')
sales = np.random.randint(200, 350, 12)  # Random sales data

df_time = pd.DataFrame({'Date': time, 'Sales': sales})

# Line chart for simulated sales data
plt.figure(figsize=(10,6))
plt.plot(df_time['Date'], df_time['Sales'], marker='o', color='b', linestyle='-', label='Sales')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar Chart (Average Petal Length per Species)
plt.figure(figsize=(10,6))
plt.bar(df['species'].value_counts().index, df.groupby('species')['petal length (cm)'].mean(), color='orange')
plt.title('Average Petal Length Per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# 3. Histogram (Distribution of Petal Width)
plt.figure(figsize=(10,6))
plt.hist(df['petal width (cm)'], bins=15, color='purple', edgecolor='black')
plt.title('Distribution of Petal Width')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter Plot (Sepal Length vs Petal Length)
plt.figure(figsize=(10,6))
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], c=df['species'], cmap='viridis', edgecolors='k')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.colorbar(label='Species')
plt.show()

# Step 4: Error Handling for Missing Libraries
try:
    import seaborn as sns
except ModuleNotFoundError:
    print("\nSeaborn library not found. Installing it...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns

# Display a message indicating completion
print("\nData analysis and visualizations complete!")
