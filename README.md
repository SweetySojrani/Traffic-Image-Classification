# Traffic-Image-Classification
Traffic Image Classification using Machine Learning

## Introduction:
### The objectives of this assignment are the following:
1. Experiment with different image feature extraction techniques.  
2. Try dimensionality reduction techniques.
3. Experiment with various classification models.  
4. Think about dealing with imbalanced data. 

## Implement Image Extraction techniques:
1. I implemented Hog Features and Hist features and compared f1 score and computation time for both. I finally chose Hist features after comparing the computation time and f1 score.
2. For Hog features, the original image(img 1) -> reduced size (img 2) -> gray image(img 3) -> edged image(img 4) -> feature vector. The feature vector was then used by dimensionality reduction as input for feature selection.

https://github.com/SweetySojrani/Traffic-Image-Classification/blob/master/Images/Original_image.PNG
https://github.com/SweetySojrani/Traffic-Image-Classification/blob/master/Images/Edged_image.PNG
https://github.com/SweetySojrani/Traffic-Image-Classification/blob/master/Images/gray_image.PNG

3. For Hist features: Used openCV to extract original image -> Hist Features(Img 5) -> Hist flattened normalized features. The normalized features were then used as input by PCA.

https://github.com/SweetySojrani/Traffic-Image-Classification/blob/master/Images/Hist_features.PNG


## Dimensionality reduction technique:
1. Implemented PCA technique on 256 Hist features of the training and test images.
And reduced the features from 256 to 25. 
2. PCA parameter of n_components=25 was selected after testing on the small data and chose the least c_component for which total F1 score remain unaffected with 256 features. However, PCA increased the total program execution time by 40%.

### Classification model: 
The small data directory has training data and test data with labels. I trained a classifier model on small training data and used the test data for validating and evaluating the model basis the f1 score generated as below: I experimented with KNN classifier model and SVM model and compared their f1 score and computation time for different parameters. I finally went ahead with KNN classifier model with k=4 for the model was less complex, less execution time and I got decent f1 score.

### F1 Score(on 50%): 0.8778

### Conclusion: 
The image features were extracted finally using Color Histogram(Hist) 256 features. PCA technique was applied to the extracted features to reduce the features extracted so as to speed up the training process of 100,000 images. The image features were then trained by KNN classifier model with parameter k=4 which gave the optimum f1 score. 
