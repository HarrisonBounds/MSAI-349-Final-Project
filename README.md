# MSAI-349-Final-Project

### Task Decription
The task for this final project is to create a Convolutional Neural Network that can identify sketches from the user. The user will first draw the sketch using a digital canvas. This iamge will then be used as the input to the CNN. The goal is provide a probability that predicts the sketch the user has drawn. This is a classification task, where each labeled sketch the model is trained on represents a different class.

### Dataset
The dataset for this project is provided by Kaggle and is called "Sketch". It is a dataset with ~20,000 sketches distriburted over 250 object categories. 

### Project Execution

**Machine Learning Techniques, Steps to Complete the Project**
-Build the user interface to draw your sketch. This sketch drawn by the user needs to be the same dimensions as the sketches in the dataset

-Create the Convolutional Neural Network to to take in the sketches from the dataset and predict what category it belongs to via probability.

-Train the model on the training set, which is a parition from the total dataset

-Predict correctly (Hopefully)!

**Interpretation of Analysis and Results**
-We will interpret out results via a cross-entropy loss function built into the model. This loss function will tell us the error "distance" between the predicted sketch and the actual label. We will then plot this on a Loss vs Number of Epochs graph to easily interpret how our model is learning.
