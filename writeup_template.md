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

[//]: # "Image References"
[image1]: ./examples/visualization.png     "Visualization"
[image2]: ./examples/grayscale.jpg         "Grayscaling"
[image3]: ./random_images/50.jpeg          "Traffic Sign 1"
[image4]: ./random_images/120.jpeg         "Traffic Sign 2"
[image5]: ./random_images/Arterial.jpg     "Traffic Sign 3"
[image6]: ./random_images/Do-Not-Enter.jpg "Traffic Sign 4"
[image7]: ./random_images/sign_1.jpg       "Traffic Sign 5"
[image8]: ./random_images/stop.jpg         "Traffic Sign 6"
[image9]: ./examples/results_recognition.png "Recognition Results"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing which data we have in the set and to which classes it corespond.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduce the size of the data set and reduce illuminance on the image,that help us to better recognize the shapes of the image.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
Data normalisation helps us with training optimisation. I normalized the image data because images can have different size, so i reshape it to one size ( 32 x 32 x 1)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:
'''
Model Architecture

# Layer 1: Convolutional 
    Input = 32x32x1
    Output = 30x30x32

# Layer 2: Convolutional
    Input = 30x30x32
    Output = 28x28x32

Pooling. 
    Input = 28x28x32
    Output = 14x14x32

# Layer 3: Convolutional
    Input = 14x14x32 
    Output = 12x12x64

# Layer 4: Convolutional
    Input = 12x12x64 
    Output = 10x10x64

Pooling
    Input = 10x10x64
    Output = 5x5x64

# Layer 5: Convolutional
    Input = 5x5x64
    Output = 3x3x128

Flatten
    Input = 3x3x128
    Output = 1152

# Layer 6: Fully Connected
    Input = 1152
    Output = 1024

# Layer 7: Fully Connected
    Input = 1024
    Output = 1024

Dropout (0.65)

# Layer 8: Fully Connected
    Input = 1024
    Output = 43
'''

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used hyperparameters:
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001

For training model I choose Adam Optimizer, because it's one of the most efficient optimizer. It works by computing adaptive learning rates for each parameter.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 0.962
* test set accuracy of 0.940

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
The first architecture was LeNet, but accuracy wasn't so high and not increased more than 80%.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

In my NN i used max pooling which help ro reduce dimensionality of the image and prevent overfiting. Next layer dropout which aslo help to prevent overfiting by "drop out" random set of activations in that layer by setting them to zero.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|     Image     |  Prediction   |
| :-----------: | :-----------: |
|   No entry   |   No entry   |
|   General caution     |    General caution    |
|     Priority road     |     Priority road     |
|   120 km/h    |  20 km/h  |
| 50 km/h | 50 km/h |
| Stop | Stop |

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This compares favorably to the accuracy on the test set of 96.2%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

One of the image was hard to recognize 120 km/h because there are classes that looks similar. One of the solution it's to add more data with this class.

![alt text][image9]

