# Payment_Delay_Project
Payment Delay Prediction Project  This project focuses on predicting whether a customer will delay payment based on account and usage data. Using a combination of data preprocessing, feature engineering, and multiple machine learning algorithms, the model identifies patterns that lead to delayed payments.

## Business Problem

A telecommunications company provides phone services to a large number of customers.  
Each customer generates a monthly bill based on their subscription plan, usage behavior, and additional services.

In real-world scenarios, not all customers pay their bills on time. Payment delays can lead to:

- cash flow issues  
- additional operational costs (reminders, customer support, penalties)  
- increased financial risk  

The company wants to answer the following question:

**"Can we predict in advance which customers are likely to delay their payment?"**

Using historical customer data, the goal is to build a machine learning model that can identify customers with a high risk of payment delay, allowing the company to take proactive actions before the problem occurs.

## Business Impact

A reliable payment delay prediction model can bring significant business value by:

- reducing financial losses caused by late payments  
- improving cash flow stability  
- optimizing customer support efforts by focusing on high-risk customers  
- enabling proactive communication strategies (reminders, flexible payment options)  
- increasing overall customer satisfaction through preventive interventions  

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

## Top 5 Factors Influencing Payment Delay

Based on exploratory data analysis (EDA), the following factors have the strongest influence on payment delays:

### 1. Number of Customer Service Calls
This is the strongest behavioral predictor. Customers who contact customer service more frequently are significantly more likely to delay payments. This reflects customer dissatisfaction, billing issues, or disputes, which directly increase the risk of delayed payments.

### 2. International Plan
Customers with an international calling plan show a much higher rate of payment delay compared to those without it. The delay rate is approximately three times higher for customers with international plans, likely due to higher and less predictable billing costs.

### 3. Total Day Minutes
Customers who spend more minutes on daytime calls tend to delay payments more often. Higher daytime usage is associated with higher monthly bills, increasing financial pressure and the likelihood of late payments.

### 4. Total Day Charge
This variable is strongly correlated with total day minutes and represents the direct financial cost of daytime usage. Higher daily charges significantly increase the probability of payment delays.

### 5. Total International Minutes
International call usage also contributes to delayed payments. Although the correlation is weaker than for daytime usage, international minutes represent costly and often unexpected charges, which increase the risk of late payments.

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

## Business Insights
The results show that payment delays are mainly driven by customer behavior and usage costs. Customers who frequently contact customer service and those subscribed to international plans present the highest risk of delayed payments, indicating potential dissatisfaction and higher billing uncertainty. Additionally, heavy daytime and international usage leads to increased charges, which significantly raises the probability of late payments. From a business perspective, these insights allow the company to proactively identify high-risk customers, apply targeted retention strategies, optimize billing communication, and design early intervention policies (such as payment reminders or personalized offers) to reduce financial risk and improve cash flow stability.

## Technologies Used
- Python  
- Pandas & NumPy (data manipulation)  
- Matplotlib, Seaborn (visualization)  
- Scikit-learn (preprocessing, scaling, modeling)  
- XGBoost, LightGBM (advanced boosting models)  
- Imbalanced-learn (SMOTE & undersampling)
