import cv2
import matplotlib.pyplot as plt

data = cv2.imread("olivettifaces.jpg")
data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
faces = {}
label = 0
count = 1
bin_list = []
for row in range(20):
    for column in range(20):
        bin_list.append(data[row*57:(row+1)*57,column*47:(column+1)*47])
        if count % 10 == 0 and count != 0:
            faces[label] = bin_list
            label += 1
            bin_list = []
        count += 1

plt.subplot(3,2,1)
plt.imshow(faces[0][0],cmap="gray")
plt.subplot(3,2,2)
plt.imshow(faces[0][9],cmap="gray")
plt.subplot(3,2,3)
plt.imshow(faces[3][0],cmap="gray")
plt.subplot(3,2,4)
plt.imshow(faces[3][9])
plt.subplot(3,2,5)
plt.imshow(faces[19][9])
plt.subplot(3,2,6)
plt.imshow(faces[19][0])
plt.show()
