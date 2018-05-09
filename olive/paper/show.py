import cv2
import matplotlib.pyplot as plt

pic = []
position = 1
for i in range(3):
    images = []
    for j in range(3):
        img = cv2.imread("olive" + str(i) + "_" + str(j) + ".jpg")
        print(img)
        # images.append(img)
        plt.subplot(3,3,position)
        plt.imshow(img)
        plt.axis("off")
        if i == 0:
            if j == 0:
                plt.title("traning dataset")
            elif j == 1:
                plt.title("test dataset A")
            else:
                plt.title("test dataset B")
        position += 1
    # pic.append(images)

#print(len(pic))


#plt.suptitle("test dataset demo")

plt.show()