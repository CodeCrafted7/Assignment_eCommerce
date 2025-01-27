# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Step 1: Merge datasets
merged_data = pd.merge(transactions, customers, on='CustomerID', how='left')
merged_data = pd.merge(merged_data, products, on='ProductID', how='left')

# Step 2: Feature Engineering
# Create customer profiles
customer_profiles = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',  # Total spending
    'TransactionID': 'count',  # Number of transactions
    'Category': lambda x: x.mode()[0],  # Favorite category
    'Region': 'first'  # Region
}).reset_index()

# One-hot encode categorical features
customer_profiles = pd.get_dummies(customer_profiles, columns=['Category', 'Region'])

# Normalize numerical features
scaler = MinMaxScaler()
customer_profiles[['TotalValue', 'TransactionID']] = scaler.fit_transform(
    customer_profiles[['TotalValue', 'TransactionID']]
)

# Step 3: Calculate Similarity
# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(customer_profiles.iloc[:, 1:])

# Step 4: Recommend Similar Customers
def get_top_similar_customers(customer_id, similarity_matrix, top_n=3):
    # Get the index of the customer
    customer_index = customer_profiles[customer_profiles['CustomerID'] == customer_id].index[0]
    
    # Get similarity scores for the customer
    similarity_scores = list(enumerate(similarity_matrix[customer_index]))
    
    # Sort by similarity score (descending order)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the customer itself and get top N similar customers
    top_similar = similarity_scores[1:top_n + 1]
    
    # Get CustomerID and similarity score
    top_similar_customers = []
    for index, score in top_similar:
        similar_customer_id = customer_profiles.iloc[index]['CustomerID']
        top_similar_customers.append((similar_customer_id, round(score, 2)))
    
    return top_similar_customers

# Step 5: Save Recommendations for the First 20 Customers
lookalike_map = {}
for customer_id in customer_profiles['CustomerID'][:20]:  # First 20 customers
    similar_customers = get_top_similar_customers(customer_id, similarity_matrix)
    lookalike_map[customer_id] = similar_customers

# Convert the map to a DataFrame
lookalike_df = pd.DataFrame(list(lookalike_map.items()), columns=['CustomerID', 'SimilarCustomers'])
lookalike_df.to_csv('Shaishav_Gupta_Lookalike.csv', index=False)

# Print the first few rows of the Lookalike.csv
print(lookalike_df.head())