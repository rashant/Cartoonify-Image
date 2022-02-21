import cv2
import numpy as np
path = 'img1.jpg'

# Reading an image in grayscale mode
image = cv2.imread(path, 1)
def edge(img):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 19)
    return edges
    # Naming a window
edges=edge(image)
cv2.namedWindow("Resize", cv2.WINDOW_NORMAL)
    # Using resizeWindow()
cv2.resizeWindow("Resize", 700, 500)
    # Displaying the image
cv2.imshow("Resize", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#edge(image)

def color_quatization(img,k):
    data=np.float32(img).reshape((-1,3))
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,0.001)
    ret,label,center=cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    result=center[label.flatten()]
    result=result.reshape(img.shape)
    return result

blurred = cv2.bilateralFilter(image, d=40, sigmaColor=500000000,sigmaSpace=500000000)

cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
cv2.namedWindow("Cartoon", cv2.WINDOW_NORMAL)
# Using resizeWindow()
cv2.resizeWindow("Cartoon", 700, 500)
# Displaying the image
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()