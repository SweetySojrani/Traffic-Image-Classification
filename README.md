# Traffic-Image-Classification
Traffic Image Classification using Machine Learning

## Introduction:
### The objectives of this assignment are the following:
1. Experiment with different image feature extraction techniques.  
2. Try dimensionality reduction techniques.
3. Experiment with various classification models.  
4. Think about dealing with imbalanced data. 

## Implement Image Extraction techniques:
I implemented Hog Features and Hist features and compared f1 score and computation time for both. I finally chose Hist features after comparing the computation time and f1 score.
For Hog features, the original image(img 1) -> reduced size (img 2) -> gray image(img 3) -> edged image(img 4) -> feature vector. The feature vector was then used by dimensionality reduction as input for feature selection.
