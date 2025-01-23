# Loan-Approval-Prediction-Model 📈

## Overview 🔍
This notebook aims to predict **loan approval status** based on various applicant details using machine learning techniques. It involves data preprocessing, exploratory data analysis (EDA), model training, evaluation, and optimization. 

---

### Features 🛠️
- **Data Cleaning and Preprocessing:** Handling missing values, encoding categorical variables, and feature scaling. 
- **Exploratory Data Analysis (EDA):** Statistical summaries and visualizations for insights. 
- **Model Building and Evaluation:** Training and testing multiple machine learning models. 
- **Ensemble Learning Approach:** Combining predictions from Decision Trees, Random Forest, and XGBoost using a Voting Classifier for improved accuracy. 
- **Performance Metrics:** Evaluates models using accuracy, precision, recall, F1-score, and confusion matrices.

---

### Dataset 📂
The dataset used for this project includes features such as: 
- **Applicant Income** - Income of the loan applicant.
- **Coapplicant Income** - Income of co-applicant (if any).
- **Loan Amount** - Loan amount requested.
- **Credit History** - Past credit performance.
- **Property Area** - Urban, Semi-Urban, or Rural.
- **Loan Status** - Target variable (Approved or Not Approved).

---

### Models Used 🤖
1. **Logistic Regression** - Baseline linear classifier.
2. **Decision Tree Classifier** - Achieved 97% accuracy.
3. **Random Forest Classifier** - Improved ensemble method with 97% accuracy.
4. **Support Vector Machine (SVM)** - Explored for performance but not selected as top.
5. **K-Nearest Neighbors (KNN)** - Tested but less effective than ensemble methods.
6. **XGBoost Classifier** - Gradient boosting model with 98% accuracy.
7. **Ensemble Model (Voting Classifier)** - Combines the top 3 models (Decision Tree, Random Forest, and XGBoost) for robustness and achieves higher stability. 

---

### Results 🏅
- The **XGBoost classifier** showed the highest individual accuracy (98%). 
- An **ensemble approach** combining the strengths of Decision Tree, Random Forest, and XGBoost improved robustness. 
- Evaluation metrics highlight the ensemble’s ability to handle misclassifications effectively. 

---


### Future Scope 🔭
- **Hyperparameter Tuning:** Optimize parameters for better performance. 
- **Additional Models:** Explore neural networks and deep learning approaches. 
- **Deployment:** Convert the notebook into a web application using Flask or Streamlit.
