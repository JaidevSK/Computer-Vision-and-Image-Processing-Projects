#Initializing the KNN model imported from sklearn
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)


#Plotting the training images
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
plt.subplot(1,4,1)
plt.imshow(Image.open('/content/mango1.jpg'))

plt.subplot(1,4,2)
plt.imshow(Image.open('/content/mango2.jpg'))

plt.subplot(1,4,3)
plt.imshow(Image.open('/content/mango3.jpg'))

plt.subplot(1,4,4)
plt.imshow(Image.open('/content/mango4.jpg'))
plt.suptitle("Training Set for Mangoes")
plt.show()

plt.figure(figsize=(20,5))
plt.subplot(1,4,1)
plt.imshow(Image.open('/content/apple1.jpg'))

plt.subplot(1,4,2)
plt.imshow(Image.open('/content/apple2.jpg'))

plt.subplot(1,4,3)
plt.imshow(Image.open('/content/apple3.webp'))

plt.subplot(1,4,4)
plt.imshow(Image.open('/content/apple4.jpg'))
plt.suptitle("Training Set for Apples")
plt.show()


#Converting the images to flattened numpy arrays
mango1 = (np.array(Image.open('/content/mango1.jpg')).flatten())
mango2 = (np.array(Image.open('/content/mango2.jpg')).flatten())
mango3 = (np.array(Image.open('/content/mango3.jpg')).flatten())
mango4 = (np.array(Image.open('/content/mango4.jpg')).flatten())
apple1 = (np.array(Image.open('/content/apple1.jpg')).flatten())
apple2 = (np.array(Image.open('/content/apple2.jpg')).flatten())
apple3 = (np.array(Image.open('/content/apple3.webp')).flatten())
apple4 = (np.array(Image.open('/content/apple4.jpg')).flatten())

#Training the model with the images and the respective labels
xtrain=np.array([mango1, mango2, mango3, mango4, apple1, apple2, apple3, apple4])
ytrain=np.array(["M", "M", "M", "M", "A", "A", "A", "A"], dtype=str)
model.fit(xtrain, ytrain)

#Testing the model and converting the output to a readable form
test_mango = (np.array(Image.open('/content/test_mango.jpg')).flatten())
test_apple = (np.array(Image.open('/content/test_apple.jpg')).flatten())
xtest=[test_mango, test_apple]
y_predicted = model.predict(xtest)

y_predicted=list(y_predicted)
for i in range(len(y_predicted)):
  if y_predicted[i]=="M":
    y_predicted[i]="Mango"
  else:
    y_predicted[i]="Apple"
print(y_predicted)


#Plotting the test images with the corresponding labels
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.imshow(Image.open('/content/test_mango.jpg'))
plt.title(f"Predicted as {y_predicted[0]}")
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(Image.open('/content/test_apple.jpg'))
plt.title(f"Predicted as {y_predicted[1]}")
plt.show()
