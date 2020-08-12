import cv2
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
imim = cv2.imread('fig.tif', cv2.IMREAD_GRAYSCALE)
print(np.shape(imim))
filt = pd.read_csv('filtercoef.csv', sep=',', header=None).to_numpy()
print(filt)
imc = signal.convolve2d(imim,filt)
plt.imshow(imc,cmap='gray')
plt.show()

