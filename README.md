# Overview of the Analysis
The purpose of this analysis was to create a deep learning model using a neural network to predict the success of funding applicants for Alphabet Soup. This organisation supports various charitable organisations. The model aimed to process and analyse a dataset containing multiple features related to charitable applicants to predict whether funding would be successful.
# Technology
-	Mach Machine Learning
-	Neural networks and deep learning 
-	Jupyter Notebook
-	Python
-	Pandas
-	NumPy
-	TensorFlow
-	Scikit-learn

# Results
## Data Preprocessing
**Target Variable(s):** The target variable for the model was the IS_SUCCESSFUL column, indicating whether an applicant's funding was successful.
**Feature Variable(s):** The features for the model included all columns except IS_SUCCESSFUL, EIN, and NAME, which were removed due to their non-beneficial nature in predicting success.
**Removed Variable(s):** The columns EIN and NAME were removed as they didn't contribute to predicting funding success.
## Compiling, Training, and Evaluating the Model
**Neurons, Layers, and Activation Functions:** The neural network model was structured with an input layer of the same dimensionality as the number of features, followed by hidden layers with 80 and 30 neurons, respectively, using ReLU activation functions. The output layer had a single neuron using a sigmoid activation function to predict success or failure.

**Target Model Performance:** The model aimed to achieve high accuracy in predicting funding success, and while it provided valuable insights, it might have room for improvement based on the application's specific needs.

**Steps for Performance Improvement:** Adjustments in the number of neurons and layers, changes in activation functions, and experimenting with different optimisation algorithms could improve the model's performance.

# Summary
The deep learning neural network model showcased moderate success in predicting funding outcomes for charitable applicants. For enhanced accuracy, further fine-tuning of hyperparameters, exploring different architectures, or considering ensemble models might lead to better performance. Alternatively, another model type, like a Gradient Boosting Machine or Random Forest, could solve this classification problem more effectively, given their robustness in handling complex datasets and non-linear relationships between features and targets.

