from scipy.cluster.vq import kmeans, vq
import numpy as np
from matplotlib import pyplot as plt

# generate data
data = np.random.rand(150, 2) # 150 elements with 2 coordinates (x and y) at columns 0 and 1 respectively
print(f"First 5 elements: \n{data[:5]}")

# compute K-means with 2 centroids (k=2)
# returns coordinated of 2 centroids and distortion (the sum of the squared distances)
centroids, _ = kmeans(data, 2)
print(f" centroids: \n{centroids}")

# assign each element to a cluster
clusters, _ = vq(data, centroids)
print(f"Each element assigned to a cluster 0 or 1: \n{clusters}")

# using logical indexing, find (x, y) coordinates of elements which belong to cluster 0 and cluster 1
# that is, for example, in case x0 = data[clusters == 0, 0], for each row (element) belonging to cluster 0 get column 0 which is x
# that is, for example, in case y0 = data[clusters == 0, 1], for each row (element) belonging to cluster 0 get column 1 which is y
x0 = data[clusters == 0, 0]
y0 = data[clusters == 0, 1]
x1 = data[clusters == 1, 0]
y1 = data[clusters == 1, 1]

# plot results and make the circles 'o' of a color, 'r' for red, 'b' for blue
# other notations: '^' triangles, 'x' crosses, 's' squares
plt.plot(x0, y0, 'or',
         x1, y1, 'ob')

# plot all centroids, 0 column is x and 1 column is y
# change the size of the marker
# marker is a green square ('sq')
plt.plot(centroids[:, 0], centroids[:, 1], 'og', markersize=20)

plt.show()