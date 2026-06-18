# Heart Disease Classification using Machine Learning


## Project Description
This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build machine learning models capable of predicting whether or not someone has heart disease based on some medical attributes. The goal is that based on some clinical attributes by a patient if we can predict whether or not he/she has heart disease.

## Dataset 
The dataset that has been used in this project is the Heart Disease dataset from Kaggle/UCI repository: https://archive.ics.uci.edu/dataset/45/heart+disease

The dataset has some attributes about the patient. The machine learning models used on this project will be trained up on those attributes and their labels in order to create patterns that will help us predict whether or not someone has heart disease.

Those attibutes are:
| Feature | Description |
|--------|-------------|
| age | Age of patient |
| sex | Gender |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Max heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of ST segment |
| ca | Major vessels |
| thal | Thalassemia |
| target | Heart disease presence |

## Technologies used
- Python
- Jupyter Notebook
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

## Project Workflow

1. Data Preprocessing
2. Exploratory Data Analysis(EDA)
3. Modelling
4. Hyperparameter Tuning
5. Evaluation
6. Feature Importance

## Models Used
- Logistic Regression
- Random Forest Classifier
- K-Neighbors Classifier

## Results
<img width="775" height="570" alt="image" src="https://github.com/user-attachments/assets/f4b37428-bb2e-4924-9725-753c31b2f328" />

The ROC curve demonstrates excellent classification performance for both models, with their curves pushed closely toward the top-left corner (the ideal classifier zone).

* **Logistic Regression (AUC = 0.93):** Outperforms the Random Forest model by a slight margin, demonstrating a 93% probability of correctly distinguishing between the positive and negative classes.
* **Random Forest (AUC = 0.92):** Shows highly competitive performance, close to the Logistic Regression model.
* **Baseline (Random Guess):** The red dashed line represents a random classifier (AUC = 0.50), which both models significantly outperform.

**Conclusion:** Both models exhibit strong discriminative power, with **Logistic Regression** being the marginally preferred choice for this specific dataset based on the AUC metric.

<img width="817" height="397" alt="image" src="https://github.com/user-attachments/assets/5c5cb015-1f86-482c-98f7-4cce496cd53a" />

* **Logistic Regression:**
  * **True Negatives (0,0):** 25 correctly classified as Class 0.
  * **True Positives (1,1):** 29 correctly classified as Class 1.
  * **False Positives (0,1):** 4 misclassified as Class 1.
  * **False Negatives (1,0):** 3 misclassified as Class 0.
* **Random Forest:**
  * **True Negatives (0,0):** 24 correctly classified as Class 0.
  * **True Positives (1,1):** 29 correctly classified as Class 1.
  * **False Positives (0,1):** 5 misclassified as Class 1.
  * **False Negatives (1,0):** 3 misclassified as Class 0.

**Key Takeaway:** Both models perform exceptionally well and score the exact same number of True Positives (29) and False Negatives (3). However, **Logistic Regression** minimizes Type I errors slightly better, making 1 less False Positive prediction than the Random Forest model. 

<img width="1342" height="721" alt="image" src="https://github.com/user-attachments/assets/4db15714-8230-4ec3-84a7-a46c9e38a04b" />

### Cross-Validated Performance Metrics for Logistic Regression

The bar chart illustrates the model's performance across three key metrics evaluated via cross-validation:

* **Recall (0.861):** The highest-scoring metric, indicating strong capability in identifying positive instances.
* **F1-score (0.809):** Demonstrates a well-balanced precision and recall profile.
* **Accuracy (0.779):** Shows a reliable overall classification correctness.

*Note: Metrics are scaled between 0.0 and 1.0.*

## Evaluation Metrics
- Confusion Matrix
- k-fold Cross Validation
- Precision
- Recaall
- Accuracy
- F1-score
- Classification report
- ROC curve/AUC

## 🚀 How to Run

1. Clone the repository
git clone https://github.com/Argyrhspet1/Heart-Disease-Prediction-using-Machine-Learning.git  
cd Heart-Disease-Prediction-using-Machine-Learning
code .

3. Install dependencies
pip install requirements.txt

4. Run the project
python src/main.py  
