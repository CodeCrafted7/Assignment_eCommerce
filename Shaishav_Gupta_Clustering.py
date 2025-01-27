# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load datasets
# Ensure the CSV files are in the same directory as this script, or provide the full path
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")
products = pd.read_csv("Products.csv")

# Merge datasets to create a unified dataset
merged_data = pd.merge(transactions, customers, on="CustomerID")
merged_data = pd.merge(merged_data, products, on="ProductID")

# Feature Engineering
# Create features for clustering
customer_features = merged_data.groupby("CustomerID").agg(
    total_spent=("TotalValue", "sum"),  # Total amount spent by each customer
    avg_transaction_value=("TotalValue", "mean"),  # Average transaction value
    total_transactions=("TransactionID", "nunique"),  # Total number of transactions
    favorite_category=("Category", lambda x: x.mode()[0]),  # Most purchased category
    signup_age=("SignupDate", lambda x: (pd.to_datetime("today") - pd.to_datetime(x)).dt.days.mean())  # Days since signup
).reset_index()

# One-hot encode categorical features
customer_features = pd.get_dummies(customer_features, columns=["favorite_category"], drop_first=True)

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features.drop("CustomerID", axis=1))

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# Choose the optimal number of clusters (e.g., k=4)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
customer_features["Cluster"] = kmeans.fit_predict(scaled_features)

# Evaluate clustering performance
db_index = davies_bouldin_score(scaled_features, customer_features["Cluster"])
silhouette_avg = silhouette_score(scaled_features, customer_features["Cluster"])

print(f"Davies-Bouldin Index: {db_index}")
print(f"Silhouette Score: {silhouette_avg}")

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
customer_features["PCA1"] = pca_result[:, 0]
customer_features["PCA2"] = pca_result[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=customer_features, palette="viridis", s=100)
plt.title("Customer Segmentation using PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

# Save the clustered data to a CSV file
output_file = "Shaishav_Gupta_Clustering.csv"
customer_features.to_csv(output_file, index=False)

# Print the path where the file is saved
print(f"Clustered data saved to: {os.path.abspath(output_file)}")


