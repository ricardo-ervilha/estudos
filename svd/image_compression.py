import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("svd\data\dog.jpg")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

U, S, V = np.linalg.svd(gray_image, full_matrices=True)
S = np.diag(S)

for r in (5, 20, 100):
    Xapprox = U[:,:r] @ S[0:r,:r] @ V[:r,:] #Aproximação
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))
    plt.show()