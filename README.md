# Vehicle detection

The goal of this project is to develop a pipeline for vehicle detection on the road.

The solution slides through the region of interest of the image and uses pre-trained xgboost model to identify windows where cars appear.

When applied to a video, the solution utilizes historical data to enforce vehicle detection.


This project uses [Advanced Lane Finding Project](https://github.com/alarionov/CarND-Advanced-Lane-Lines) for lane line detection.

Research of the problem can be found in [this notebook](https://github.com/alarionov/CarND-Vehicle-Detection/blob/master/Vehicle%20Detection.ipynb).
Through the description of the pipeline, code references to particular cells of research notebook with implementation will be made.

[Video result](https://www.youtube.com/watch?v=IDvh9KkcZEM)

## Model Training

The most crucial part of our solution is a model, which will predict if a window contains a car.

For this purpose [XGBoost](http://xgboost.readthedocs.io/en/latest/) was selected since it has a very good reputation for solving classification problems.

### Data

All our images have a size of 64x64 pixels and 3 rgb channels. `[cell#5]`

Images are stored in PNG files, so every pixel is a float value in a range from `0` to `1` (during video processing we will work with JPEG images, so we should keep in mind that we need to scale them properly).

All images are split into separate directories for vehicles and nonvehicles.

We will put paths for those files into separate dataset and for each dataset we will set a label: `auto=1` for vehicles and `auto=0` for non vehicles.`[cells#2-3]`

Then we will take 8000 images from each dataset and combine it into our final dataset `data`.`[cell#4]`

Car example             |  Non car example
:----------------------:|:------------------------------:
![car](examples/car.png)|![non car](examples/non_car.png)

### Feature Extraction

To train our model we will extract following features from every image:

* Histogram of Oriented Gradients
* Color Histogram of RGB image
* Color Histogram of LAB image
* Color Histogram of LUV image
* Color Histogram of HSV image

`[cells#7-8]`

#### Histogram of Oriented Gradients

[Histogram of Oriented Gradients (HOG)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) is a feature extraction method which will allow us to catch a structure of a car on the image.

![hog example](examples/HOG_example.png)

#### Color Histogram

We will also use channels from different color schemes to find useful features for vehicle detection.

### Feature Selection

Feature extraction gives us 3300 features. This number of features will significantly increase training time of our model, it also can lead to overfitting and worse performance if we have a lot of useless features in our model.

To decrease the number of features we will remove features with low variance - features, which have very similar values for both vehicles and non vehicles.

`[cell#9]`

### Feature Scaling

While feature extraction we combined features of different nature, so it's very possible that their scales will be different.

It might cause that our model pays more attention to features of bigger scale. Scaling our features will help us to avoit it.

`[cell#10]`

### Training

After our data is ready, we can split it into a training set(0.44), a validation set(0.22) and a test set(0.33).

We will use the training set for training the model, the validation set to calculate an error after every round and the test set to calculate the real error for our model after the training.

`[cells#11-19]`

## Window sliding

With the trained model we can start analysing images of the road.

Size of vehicles on video images will depend on the relative position of the camera.

Vehicles, which are closer to the camera, will look bigger and vice versa.

To detect vehicels efficiently we will use different sizes of sliding window for differnt regions of the image.

We will use 3 different window sizes: 64x64, 96x96 and 128x128 for 3 different regions of the image.

Since we trained our model on images of size of 64x64, we need to resize a content of our sliding window to the same size.

64x64 window                              | 96x96 window                             | 128x128 window
:----------------------------------------:|:----------------------------------------:|:--------------------------------------------:
![64x64 window](examples/64x64_window.png)|![96x96 window](examples/96x96_window.png)|![128x128 window](examples/128x128_window.png)

## Classification

After we get images from regions we are interested in, we combine them into a dataset and use the same feature extraction, feature selection and feature scaling we used for the training data.

Now we are ready to classify every piece of image and predict a probability whether it's a car.

`[cell#21]`

## Bondaries and Heatmap

With probabilities we calculated above we can add `1` to every pixel of every window which contains a car (probability > 0.5). `[cell#22]`

This way we will create a heatman, which will indicate areas of the image, which contain vehicles.

Now we just need to draw boundary boxes around those regions on the original image.

![heatmap](examples/heatmap.png)

## Wrong predictions

Even our model has 99% accuracy on balanced test dataset, sometimes it's not enough.

From time to time our model will detect vehicles where there is none or it won't see a car on the image.

We will use few tricks to make our detection more robust.

* set treshold for a probability very high: 0.99, to include in heatmap only regions, which very likely contain a car `[cell#22]`
* set treshold for a heatmap to remove regions with a smaller probability of a car `[cells#24-25]`
* use a previous heatmap to enforce detection in regions where a car was detected recently `[cell#25]`

## Conclusion

Though this simple pipeline gives good results already, there is a place for improvement.

### Feature Selection

In addition to filtering near zero variance features, we should consider using [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) or [LASSO regression](https://en.wikipedia.org/wiki/Lasso_(statistics)).

### Dinamic ROI

In current solution, ROI is fixed and shifted right because the car is driving by left lane with a separator on the left of the car.

To make this solution working on different kind of roads it would be beneficial to detect different types of obstacles and calculate ROI dinamically.

### Deep Learning

It also would be interesting to compare this results to what [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) could do.
