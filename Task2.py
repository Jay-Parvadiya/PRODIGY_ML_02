import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Mall_Customers.csv')
data.head()

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
for i in range(5):
    plt.scatter(X_scaled[data['Cluster'] == i, 0], X_scaled[data['Cluster'] == i, 1], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=200, label='Centroids')
plt.title('Customer Segments')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.grid(True)
plt.show()

data.to_csv("clustered_customers.csv", index=False)

# Step 1: Load test dataset
test_data = pd.read_csv('Mall_Customers_Test.csv')  # Replace with actual filename
test_data.head()

# Step 2: Select relevant features
X_test = test_data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Scale test data using same scaler as training
X_test_scaled = scaler.transform(X_test)  # Use the same scaler used for training

# Step 4: Predict cluster labels using trained KMeans model
test_clusters = kmeans.predict(X_test_scaled)
test_data['Cluster'] = test_clusters

# Step 5: Save the test dataset with cluster labels
test_data.to_csv('clustered_test_customers.csv', index=False)
print("✅ Cluster labels added and saved as 'clustered_test_customers.csv'")

