# Traffic-Sign-Detection
Built a model to train traffic signs with over 35000 images of 43 different classes with the help of Tensorflow and Keras .
Train and classify Traffic Signs using Convolutional neural networks This will be done using OPENCV in real time using a simple webcam.


Data Structure:
- labels.csv : Contains classification labels.
- Traffic_Signal_Main.py : CNN model building.
- Traffic_Test.py : To test with new data with help of webcam using cv2.


Steps to build model:
1. load data with the given labels. 
2. All images have 3 channels with different sizes. Resize images with cv2 and greyscale it.
3. Find the image distribution with bar chart.
4. Devide images in to train,validation, test images.
5. Create image generator with augmentation. 
6. Create model with different combinations of CONV2D, MAXPOOLING, DROPOUT,FLATTEN, DENSE Layer.
7. Evaluate model
8. Tune with Keras hyperparameter technique
9. Create pickle file of best performing model. Which will be input for Traffic_Test.py.
10. In Traffic.py load pickle file, open webcam and detect object using cv2.
11. Convert object to greyscale, resize as mentioned while training.
12. Give input to pickle file.
13. Display prdiction on screen.



