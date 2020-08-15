from matplotlib.image import imread 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os

PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=500):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

    image = imread(os.path.join("1.jpg"))


image.shape
X = image.reshape(-1, 3)

segmented_imgs = []
n_colors = (15, 12, 9, 6, 3)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))
    
plt.figure(figsize=(100,50))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

plt.subplot(231)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
        
    altering_cons = np.random.randint(0,255)
    plt.imshow((segmented_imgs[idx]*altering_cons).astype(np.uint8))
         
    plt.title("{} colors".format(n_clusters))
    plt.axis('off')

#save_fig('Output', tight_layout=False)
plt.show()