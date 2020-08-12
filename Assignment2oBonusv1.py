import cv2
from scipy import signal
from scipy.ndimage import gaussian_filter, laplace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
imim = cv2.imread('fig.tif', cv2.IMREAD_GRAYSCALE)
print(np.shape(imim))
imc = laplace(gaussian_filter(imim, sigma=1))
plt.imshow(imc,cmap='gray')
plt.show()

