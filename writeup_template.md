# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

### ALL IMAGES AND GRAPHS ARE TO BE REFERENCED INSIDE THE NOTEBOOK,

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### Data Download and Loading (Cell 1,2,3,4)

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually. (Cell 5,6)

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

Here, I also display the distribution of classes based on ID in text format.

#### 2. Include an exploratory visualization of the dataset. (Cell 7,8)

* Bar graph displaying the distribution of classes based on ID.
* Display of the first image of each class.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.) - Cell 9


* Initially I created the 'enhance function' in order to do a contrast filter.
* Next, I created a combined function 'apply_brightness_contrast' function which performs a brightness and contrast ratio increase which increases the intensity of the image.
* But, I found that the image values needed to be normalized so created 'image_normalizer' function to bring the values to between 0-255.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model. - (Cell 10,11,12,13,14,15,16,17)

* Keras was used due to my expertise in the library usage. experience.
* Pre-processing was performed over all train, validation and test images.
* Label Encoder was used to encode the class variables in 43 columns.
* Shape of all 3 sets were verified.
* Model with learning_rate = 0.01 was created. The model architecture is given in the notebook as model.summary()
* Function for reducing the learning_rate was created 'lr_reducer' which reduces by 10 every 10 epochs.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate. - (Cell 15)

* For training, I used the training images along with the encoded labels. 
* I used 3 callbacks: EarlyStopping, LearningRateScheduler, ModelCheckpoint.
* I gave it a 100 epoch threshold but the earlystopper gave way at epoch 12.
* Validation set was pushed directly with the training as validation_data in model.fit.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem. - (Cell 18,21,22,23)

##### The algoithm chooses the best model. So:
My final model results were:
* training set accuracy of 99.95 %
* validation set accuracy of 98.85 %
* test set accuracy of 98.13 %

##### Properties:

* I chose the Conv2D -> MaxPool -> Dropout architecture due to its success stories in image processing.
* I followed the convention of having 1 MaxPooling layer for every 2 convolutions in order to store as many features of the image as possible.
* Final Accuracy is very close to ideal conditions which prove its effectiveness.

##### Store and Load Model

* I created this part to store the model for future testing without re-training the model.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify. - (Cell 24)

Downloaded 6 images from Google Images.
* Speed limit (30km/h)
* Bumpy Road
* Ahead only
* No vehicles
* Go straight or left
* General caution

All the labelling was done using index and then referenced back to the signnames document.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric). - (Cell 28, 32)

Here are the results of the prediction:

| Image			                    |     Prediction	        					| 
|:---------------------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| Speed limit (30km/h)   						| 
| Bumpy Road     			        | Bumpy Road 									|
| Ahead only					    | Ahead only									|
| No vehicles					    | No vehicles									|
| Go straight or left	      		| Go straight or left					 		|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 98.13 %

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 32nd cell of the Ipython notebook.

For all 5 images, the model is sure abou the label, and the image does contain a stop sign. The top five soft max probabilities were

For Image 1 - Speed limit (30km/h):

| Probability         	    |     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| 1.0         			    | Speed limit (30km/h)   						| 
| 1.06e-11     				| Speed limit (50km/h) 							|
| 1.23e-13					| Road narrows on the right									|
| 2.23e-14	      			| End of speed limit (80km/h)					 		|
| 1.93e-14				    | Speed limit (70km/h)      							|

For Image 2 - Bumpy Road:

| Probability         	    |     Prediction	        				| 
|:-------------------------:|:-----------------------------------------:| 
| 0.99     				    | Bumpy Road 								|
| 9.44e-8					| Bicycles crossing							|
| 7.82e-10	      			| Slippery road					 			|
| 1.69e-10				    | Speed limit (60km/h)      				|
| 2.67e-11				    | General caution      						|

For Image 3 - Ahead only:

| Probability         	    |     Prediction	        				| 
|:-------------------------:|:-----------------------------------------:| 
| 1.0         			    | Ahead only             					| 
| 5.53e-14     				| Go straight or right 						|
| 1.28e-14					| Yield					    				|
| 1.42e-16	      			| Sped limit (30km/h)		 				|
| 1.13e-16				    | Children crossing      					|

For Image 4 - No vehicles:

| Probability         	    |     Prediction	        				| 
|:-------------------------:|:-----------------------------------------:| 
| 1.0         			    | No vehicles   					    	| 
| 0.99     				    | Yield 									|
| 1.0					    | End of all speed and passing limits		|
| 1.0	      			    | Speed limit (50km/h)						|
| 1.0				        | Traffic signals      						|

For Image 5 - Go straight or left:

| Probability         	    |     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| 1.0         			    | Go straight or left   						| 
| 1.39e-14     				| Keep right 									|
| 3.15e-15					| Keep left									    |
| 1.16e-16	      			| Sliperry road					 				|
| 1.10e-16				    | Speed limit (20km/h)      					|

The cell shows the bar graph plot for the top 5 predictions given by the model on the test images.
An additional 6th image on General Caution was also tested just for fun.

Please check the notebook for reference.


## Challenges:

* During pre-processing finding the ideal lighting and contrast conditions were a challenge.
* Finding the best hyperparamter combinations for the model architecture was also troublesome as the difference bween layers and the nodes within them were difficult to manage.
* Created the bar graph for the top 5 predictions was tricky.
* All challenges mentioned above were solved.

## Future Enhancements:
* This model even though has good test accuracy, we can try creating a more complex model.
* Data augmentation can help creating more training data.
* GAN can also contribute in creation of more training data.

