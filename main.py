import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

'''
-------------------------------------------------------------------------------
 1. Generate synthetic dataset
-------------------------------------------------------------------------------
'''
np.random.seed(42)
n_samples = 300

# Cluster 1: Routine, low-resource
cluster1 = np.column_stack([
    np.random.normal(loc=0.5, scale=0.2, size=n_samples),
    np.random.normal(loc=60, scale=15, size=n_samples),
    np.random.normal(loc=2, scale=0.5, size=n_samples),
    np.random.normal(loc=200, scale=50, size=n_samples)
])

# Cluster 2: Moderate, predictable
cluster2 = np.column_stack([
    np.random.normal(loc=4, scale=1.0, size=n_samples),
    np.random.normal(loc=120, scale=20, size=n_samples),
    np.random.normal(loc=4, scale=1.0, size=n_samples),
    np.random.normal(loc=1000, scale=200, size=n_samples)
])

# Cluster 3: Complex, high-resource
cluster3 = np.column_stack([
    np.random.normal(loc=12, scale=3, size=n_samples),
    np.random.normal(loc=200, scale=40, size=n_samples),
    np.random.normal(loc=8, scale=2, size=n_samples),
    np.random.normal(loc=5000, scale=1000, size=n_samples)
])

# Combine
X = np.vstack([cluster1, cluster2, cluster3])
df = pd.DataFrame(X, columns=["length_of_stay", "theatre_time", "staff_count", "consumables_cost"])

'''
-------------------------------------------------------------------------------
 2. Preprocess
-------------------------------------------------------------------------------
'''
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

'''
-------------------------------------------------------------------------------
 3. Determine optimum number of clusters
-------------------------------------------------------------------------------
'''
wcss = [] # within-cluster sum of squares
sil_scores = []

K_range = range(2,11) # try 2 to 7 clusters
for k in K_range:
    kmeans = KMeans(n_clusters=k,random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    
    
# Plot Elbow Method
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, marker='o')
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS (inertia)")
plt.title("Elbow Method")

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(K_range, sil_scores, marker='o')
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")

plt.tight_layout()
plt.show()

'''
-------------------------------------------------------------------------------
 4. Run final clustering with chosen k
-------------------------------------------------------------------------------
'''
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans_final.fit_predict(X_scaled)

summary = df.groupby("cluster").mean().round(1)
print(summary)


'''
-------------------------------------------------------------------------------
 5. Profile clusters
-------------------------------------------------------------------------------
'''

# Profile clusters (mean values)
cluster_profile = df.groupby('cluster').mean().round(1)

# Add cluster size and proportion of total cases
cluster_counts = df['cluster'].value_counts().sort_index()
cluster_profile['num_cases'] = cluster_counts
cluster_profile['proportion_%'] = (cluster_counts / len(df) * 100).round(1)

# Calculate total sums per feature per cluster
cluster_sums = df.groupby('cluster')[['length_of_stay','theatre_time','consumables_cost']].sum()

# Calculate proportion of total for each feature
cluster_prop = (cluster_sums / cluster_sums.sum() * 100).round(1)
cluster_prop.rename(columns=lambda x: x + '_prop_%', inplace=True)

# Merge proportions into profile
cluster_profile = cluster_profile.merge(cluster_prop, left_index=True, right_index=True)
print(cluster_sums)
print(cluster_profile)

# Export to Excel
cluster_profile.to_excel("cluster_summary.xlsx", sheet_name="Cluster Profiles")

print(cluster_profile)
print("Cluster summary with resource proportions exported to cluster_summary.xlsx")


























































