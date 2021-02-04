from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

path = "sign_data"  # folder with all the class folders
labelFile = 'labels.csv'  # file with all names of classes
batch_size_val = 50  # how many to process together
steps_per_epoch_val = 500
epochs_val = 10
imageDimesions = (32, 32, 3)
testRatio = 0.2  # if 1000 images split will 200 for testing
validationRatio = 0.2  # if 1000 images 20% of remaining 800 will be 160 for validation

# Importing images using CV2
count = 0
images = []
label = []

folders = os.listdir(path)
print("Total Classes Detected:", len(folders))

no_of_folders = len(folders)
print("Importing Classes.....")

for folder in range(0, no_of_folders):
    pic_list = os.listdir(path + "/" + str(count))
    for pic in pic_list:
        curImg = cv2.imread(path + "/" + str(count) + "/" + pic)
        images.append(curImg)
        label.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
label = np.array(label)
print("label=========>>>>>>>   ", label)

# Spilt data into train and test
X_train, X_test, y_train, y_test = train_test_split(images, label, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

print("Data Shapes")
print("Train", end="")
print(X_train.shape, y_train.shape)
print("Validation", end="")
print(X_validation.shape, y_validation.shape)
print("Test", end="")
print(X_test.shape, y_test.shape)

# READ CSV FILE
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))

'''
#Display Sample Imges
num_of_samples = []
cols = 5
num_classes = no_of_folders
fig, axs = plt.subplots(nrows = num_classes, ncols = cols, figsize = (5, 300))
fig.tight_layout()
for i in range(cols):
    for j,row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)- 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j)+ "-"+row["Name"])
            num_of_samples.append(len(x_selected))


print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()'''


# PREPROCESSING THE IMAGES

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)  # convert image to grayscale
    img = equalize(img)  # standardize lightning in images
    img = img / 255  # normalize image
    return img


X_train = np.array(list(map(preprocessing, X_train)))  # preprocess all images
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))
#cv2.imshow("GrayScale Images", X_train[random.randint(0, len(X_train) - 1)])

print("Before ====>>>>>>>>", X_train[0].shape)

# ADD A DEPTH OF 1
#X_train = X_train.reshape([X_train.shape[0], [1]] + list(X_train.shape[1:]))
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
print("After ====>>>>>>>>", X_train[0].shape)

## AUGMENTATAION OF IMAGES: TO MAKEIT MORE GENERIC
data_gen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                             shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                             rotation_range=10)  # DEGREES

data_gen.fit(X_train)

batches = data_gen.flow(X_train, y_train, batch_size=20)

X_batch, y_batch = next(batches)

# to show augmented image
'''fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
    axs[i].axis('off')
plt.show()'''

y_train      = to_categorical(y_train, no_of_folders)
y_validation = to_categorical(y_validation, no_of_folders)
y_test       = to_categorical(y_test, no_of_folders)


# Designed CNN
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)  # kernal
    size_of_Filter2 = (3, 3) # kernal2
    size_of_pool = (2, 2)  # Pooling
    no_Of_Nodes = 500  # nodes in DNN

    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(32, 32, 1),  activation='relu')))
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size = size_of_pool))  # DOES NOT EFFECT THE DEPTH/NO OF FILTERS

    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))  # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    model.add(Dense(no_of_folders, activation='softmax'))  # OUTPUT LAYER
    # COMPILE MODEL
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model



# Train Model
model = myModel()
print(model.summary())

history = model.fit_generator(data_gen.flow(X_train, y_train, batch_size = batch_size_val),
                            steps_per_epoch=steps_per_epoch_val,
                            epochs=epochs_val,
                            validation_data=(X_validation,y_validation),
                            shuffle=1)

#Plot Accuracy
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score =model.evaluate(X_test,y_test,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])


# STORE THE MODEL AS A PICKLE OBJECT
model_pickel =  open("trained_model.p","wb")  # wb = WRITE BYTE
pickle.dump(model, model_pickel)

model_pickel.close()
cv2.waitKey(0)