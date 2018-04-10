# Fruit-Image-Classifier-SVM

## Introduction

A support vector machine(SVM) is a classifier that uses a separating hyperplane to analyze and classify data. The following images visualize a 2 dimensional situation of SVM. The separating hyperlane represents as the lines. And two classes (black circles, white circles) can be seen as well in the images. The question is which one is the optimal hyperplane. The answer is to find a hyperplane that is as far as possible from all the points. To put it more formal, training SVM classifier is the problem of finding the optimal hyperplane that gives the largest minimum distance to the training data points. It turns SVM to an optimization problem.

![Validation Score](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Svm_max_sep_hyperplane_with_margin.png/220px-Svm_max_sep_hyperplane_with_margin.png)
![Validation Score](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Svm_separating_hyperplanes_%28SVG%29.svg/220px-Svm_separating_hyperplanes_%28SVG%29.svg.png)

Images source:  https://en.wikipedia.org/wiki/Support_vector_machine

## Feature Extraction

For the feature extractions, we use several feature extractors provided by the scikit image library. These extractors enable us to build the feature vectors that are later used as the input for SVM classifier. We test different features provided by scikit image features and choose those who provide the best results. The short explanation of each extractors are introduced below.

- ORB: Oriented FAST and rotated BRIEF feature detector and binary descriptor extractor.
- BRIEF: Binary Robust Independent Elementary Features is an efficient feature point descriptor. For each keypoint, intensity comparisons are carried out for a specifically distributed number N of pixel-pairs resulting in a binary descriptor of length N. For binary descriptors the Hamming distance can be used for feature matching, which leads
to lower computational cost in comparison to the L2 norm (scikit)
- CENSURE: Center Surround Extremas for Realtime Feature Detection and Matching
- Histogram of the grey scale image
- Corner peaks: Find corners in corner measure response image.
- Histogram of Oriented Gradients

An image gradient is a concept to describe the change of intensity or color in an image. In every pixel of an image, a gradient contains two values: a magnitude and a direction. A magnitude is a description about how strong is the change, while the value of a direction is used to describe the angle of the changes. There are several steps for building a HOG. First, preprocessing the image to set a certain size ratio for every images. Then, computing the horizontal and vertical gradients. Next, dividing the image pixels into cells and calculating the Histogram of Gradients in each designated cells. For example, an 128*64 pixels images can be divided into 128 8*8 cells. The x-axis of a HOG is divided by the angles of gradients and y-axis represents the magnitudes. In order to less sensitive to the overall lighting changes, we need to normalize the HOG. Finally, to compute the feature vectors, the normalized HOG is turned into one vector.

## Feature Selection Optimization

Since the different fruits that are used for classification have many different characteristics and forms, it was noticed that extracting certain features from certain fruits resulted worse or not optimal accuracy rates. For this reason, an algorithm was developed where a configuration list is created that details which feature extractors should be utilized for a certain fruits. This is accomplished by creating a list of every possible status (included/not included represented as booleans) for each extractor, thus resulting in 16 configuration lists. For example, when comparing apples to bananas extracting only the corner peaks the and the edge histogram results in a better result (90,5%; [False, False, True, True]) than if the ORB, BRIEF and edge histogram are extracted without the corner peaks (81%; [True, True, True, False]). A possible explanation for this is that some fruits lack certain characteristics or possess them in insufficient quality and quantity. To correctly estimate which features worsen model accuracy, a sample of 100 images is extracted from the each fruits training images and then fed into the SVM model. The accuracy for each configuration list is returned and finally the configuration list chosen which bears the highest one. This list is later used for determining which features need to be extracted during the main process.
It was noticed that the HOG and grayscale histogram extractors always returned more positive results if they were included which is why they are automatically used to extract their respective features and feed them into the model.

## Grid Search

As described during the theoretical introduction of the SVM classifier, two parameters need to be chosen carefully to deliver the best possible results, C and gamma. Their values can greatly influence the resulting accuracy of the model and thus a careful fine-tuning. The SkLearn machine learning library offers the Method  GridSearchCV  which calculates the best possible parameter combination for a certain parameter range. To calculate C, the values from 1.0e-02 to 1.0e10 were used. To calculate gamma, the values from 1.0e-09 to 1.0e3 were used. This process resulted in increased accuracy of up to 3% compared to the default values of gamma = 1.0e-9 and C = 10.0 which have been calculated earlier for specific differences in C and gamma, e.g. with C being 1.0 or 100.0. However, when using grid search it is important to consider that the model is at risk of being overfitted towards the used training data.

## Result of SVM Testing

By changing the parameters in SVM, we tried to find the best combination of features among every possible combination of four features: ORB, Corner peaks, Edge and CENSURE. Since the remaining two features, histogram of gray scale image and HOG, always increased prediction accuracy we decided to always include them. The test accuracy of the best combinations of features for each pair of fruits is depicted below. The best combination with optimized grid parameters for n = 400 is depicted as well as the mean accuracy of all possible combinations and the best possible combination for n = 100.

![Validation Score](https://github.com/Witsung/SVM-Fruit-Image-Classifier/blob/master/table.png)

The results show that there is no feature from the combination that was included in every best possible combination. One can also notice that the accuracy for n=100 samples varies greatly depending on if the best possible feature combination is extracted. One can also see that the best possible combination for n=100 and n=400 are not clearly superior or superior to each other. This high variance could also be caused by the relatively low sample number. However, the mean accuracy of all possible combinations is always lower than for the best possible combination for n = 400.
