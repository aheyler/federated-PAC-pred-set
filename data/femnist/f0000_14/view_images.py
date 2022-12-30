import numpy as np
import matplotlib.pyplot as plt

images = np.load("training_images.npy")
for i in range(5): 
    plt.figure()
    plt.imshow(images[i], cmap='gray')
    plt.savefig(f"image_{i}.png")