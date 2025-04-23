import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

############ Unsupervised Learning (Tax Data Insights) ############
############ K-Means Clustering ############
# Load an image
# image = io.imread('../galaxy-portfolio/public/planet/textures/earth_albedo.jpg')
# image = image / 255.0  # Normalize pixel values to [0, 1]

# # Reshape the image into a 2D array of pixels
# pixels = image.reshape(-1, 3)

# # Apply K-Means to cluster pixel colors
# kmeans = KMeans(n_clusters=8, random_state=42)  # Reduce to 8 colors
# kmeans.fit(pixels)

# # Replace each pixel with its corresponding cluster center
# compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
# compressed_image = compressed_pixels.reshape(image.shape)

# # Display the original and compressed images
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].imshow(image)
# ax[0].set_title("Original Image")
# ax[0].axis('off')

# ax[1].imshow(compressed_image)
# ax[1].set_title("Compressed Image (8 Colors)")
# ax[1].axis('off')

# plt.show()

####################################
# Sample data: [Annual Income (k$), Spending Score (1-100)]
data = np.array([
    [15, 39], [16, 81], [17, 6], [18, 77], [19, 40],
    [20, 76], [21, 6], [22, 77], [23, 40], [24, 76],
    [25, 35], [26, 99], [27, 5], [28, 95], [29, 36],
    [30, 98], [31, 5], [32, 93], [33, 38], [34, 97]
])

# Create and fit the K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters
kmeans.fit(data)

# Get cluster labels for each data point
labels = kmeans.labels_

# Get the cluster centroids
centroids = kmeans.cluster_centers_

# Visualize the clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

print("Cluster labels:\n", labels)
print("Cluster Centers:\n", centroids)

############ PCA  ############
# Sample data: Students' scores in 5 subjects
data = np.array([
    [85, 92, 88, 75, 80],
    [78, 85, 82, 70, 75],
    [90, 95, 94, 85, 88],
    [65, 70, 68, 60, 65],
    [72, 78, 75, 68, 70]
])

# Step 1: Standardize the data (important for PCA)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 2: Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Step 3: Print explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Reduced Data (2D):\n", data_pca)

# Step 4: Visualize the reduced data
plt.scatter(data_pca[:, 0], data_pca[:, 1], c='blue', label='Students')
plt.title("PCA: Students' Scores Reduced to 2 Dimensions")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()