# Payment_Delay_Project
# Payment Delay Prediction Project  This project focuses on predicting whether a customer will delay payment based on account and usage data. Using a combination of data preprocessing, feature engineering, and multiple machine learning algorithms, the model identifies patterns that lead to delayed payments.


## Project Overview
The dataset contains customer data including:

- Account information (`account_length`, `area_code`)
- Service subscriptions (`international_plan`, `voice_mail_plan`)
- Usage statistics (day, evening, night, and international calls and charges)
- Number of customer service calls
- Payment delay (`payment_delay` - target variable)

The goal of the project is to:

- Understand which factors influence payment delays
- Build predictive models to classify customers at risk
- Compare multiple machine learning algorithms and scaling techniques
- Handle imbalanced data to improve predictive performance

## Key Steps

### 1. Data Preprocessing
- Handling missing values
- Encoding categorical variables (Yes/No → 1/0)
- One-hot encoding for categorical features like `state` and `area_code`
- Applying log transformation to skewed numeric features
- Scaling numeric features using **StandardScaler** and **MinMaxScaler**

### 2. Exploratory Data Analysis (EDA)
- Distribution plots and boxplots
- Correlation heatmaps
- Countplots for target variable and key categorical features

### 3. Modeling
- Split data into train/test sets
- Tested multiple algorithms:
  - Logistic Regression
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
  - Random Forest
  - Extra Trees Classifier
  - Gradient Boosting Classifier
  - XGBoost
  - LightGBM
- Evaluated models using **accuracy**, **confusion matrix**, and **classification report**

### 4. Handling Imbalanced Data
- Oversampling using **SMOTE**
- Undersampling using **RandomUnderSampler**
- Tested model performance after balancing classes

### 5. Hyperparameter Tuning
- Grid search applied on LightGBM to optimize:
  - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- Achieved improved predictive performance for payment delay detection

### 6. Predictions
- Applied the trained and tuned model on the test dataset
- Generated final predictions for `payment_delay`

## Technologies Used
- Python  
- Pandas & NumPy (data manipulation)  
- Matplotlib, Seaborn (visualization)  
- Scikit-learn (preprocessing, scaling, modeling)  
- XGBoost, LightGBM (advanced boosting models)  
- Imbalanced-learn (SMOTE & undersampling)
