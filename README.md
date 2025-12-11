<p align="center">
  <img src="https://github.com/KRISHexe520/Credit-Risk-Classification-Model/blob/main/images/LoanBanner.png" alt="Credit Risk Banner" width="100%">
</p>

# 📘 Credit Risk Classification Model  
<p align="center">

  <!-- Python Version -->
  <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white" />

  <!-- Libraries -->
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-yellow?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-orange?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?logo=matplotlib&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-Visualization-9cf?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Models-green?logo=scikit-learn&logoColor=white" />

  <!-- ML Project Badges -->
  <img src="https://img.shields.io/badge/Project-Type%3A%20Classification-red" />
  <img src="https://img.shields.io/badge/Model-Random%20Forest%20%7C%20Logistic%20Regression-brightgreen" />
  <img src="https://img.shields.io/badge/Dataset-Imbalanced%20Data-critical" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />

  <!-- Tools -->
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub-Repository-black?logo=github" />

</p>

A Machine Learning Project for Predicting Loan Default Risk  

This project focuses on building a classification model that predicts whether a loan applicant is likely to **default** or not.  
It includes a complete end-to-end pipeline: **data cleaning, EDA, preprocessing, model building, evaluation, and insights**.

---

## 🚀 Project Overview

Credit risk analysis helps lenders estimate the probability of a borrower defaulting on a loan.  
Using this dataset, we:

- Explore and visualize customer financial behavior  
- Identify patterns in repayment vs default  
- Build ML models to predict risk  
- Evaluate model performance  
- Generate insights for business decisions  

---

## 📂 Repository Structure
```
📁 Credit-Risk-Classification-Model
│── 📄 Credit Risk Classification Model.ipynb
│── 📄 requirements.txt
│── 📄 README.md
│── 📁 images/ (EDA plots, heatmaps, etc.)
```
---

## 🧠 Objectives

- Understand major factors affecting credit risk  
- Perform complete Exploratory Data Analysis (EDA)  
- Handle missing values & outliers  
- Encode categorical variables  
- Scale numerical features  
- Train & compare ML models  
- Evaluate using accuracy, precision, recall, F1-score, ROC-AUC  

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-Learn  
- Jupyter Notebook  

---

## 🔍 Workflow Summary

### **1️⃣ Importing Libraries & Loading Dataset**
Basic setup using Pandas, NumPy, Matplotlib, Seaborn, and Sklearn.

---

### **2️⃣ Initial Data Exploration**

- Data shape  
- Data types  
- Summary statistics  
- Checking unique values  
- Understanding target variable **(Default Status)**  

---

### **3️⃣ Handling Missing Values**

- Identified missing data  
- Visualized missing patterns  
- Imputed values using mean/median/mode depending on column type  

---

### **4️⃣ Exploratory Data Analysis (EDA)**

Performed multiple analyses including:


**- 1️⃣ Boxplots of Bill Amounts (Last 6 Months)**

<img width="1200" height="600" alt="Bill Amounts (Last 6 Months)" src="https://github.com/user-attachments/assets/e96ed785-d8a5-438c-bee1-976cf5589d74" />
 
**Insight:**
Bill amounts for all 6 months follow a similar distribution.

There are many high-value outliers, indicating that some customers have extremely high outstanding balances.

Most values are concentrated near the lower range, meaning the majority of customers maintain smaller bills.

Outliers may significantly influence the model; scaling or log transformation could help stabilize variance.

**Business Meaning:**
A small group of customers carry very large bills, which may contribute to higher default risk.


**2️⃣ Target Variable Distribution (Default vs Not Default)**


<img width="500" height="400" alt="class_distribution" src="https://github.com/user-attachments/assets/ea40ed5f-669d-43d5-bb9a-bd9889ec9be4" />

**Insight:**
The dataset is highly imbalanced:

Non-defaulters ≈ 23,000

Defaulters ≈ 7,000

The majority class dominates the dataset.

Impact on Model:

Imbalanced data makes models biased toward predicting “Not Default”.

This explains why Logistic Regression completely failed to detect defaulters.


**3️⃣ Logistic Regression Confusion Matrix**


<img width="640" height="480" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/aad0e508-7969-4986-923c-5e7af4e117ab" />
 
**Insight :** 
The model predicts EVERY customer as Not Default.

It correctly predicts all actual non-defaulters (4673) but predicts 0 defaulters correctly.

This is a direct result of:

class imbalance

simple linear model assumptions

overlapping feature distributions

**Conclusion:**
Logistic Regression is not suitable for this dataset without resampling or class weighting.


**4️⃣ Correlation Heatmap**


<img width="1200" height="800" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/98713d19-17a8-456a-a266-7233bec11eb5" />

**Insight:**
Strong positive correlations among BILL_AMT1–BILL_AMT6, meaning customers' bill amounts remain consistent month-to-month.

Strong correlations among PAY_AMT1–PAY_AMT6, indicating consistent payment behavior.

Limit balance (LIMIT_BAL) has moderate correlation with bill amounts—customers with higher limits usually have larger bills.

PAY status variables (delay indicators) show weak-to-moderate correlation with payment amounts.

**Business Learning:**
Billing and payment patterns are consistent over time → customer behavior is predictable, useful for modeling.


**5️⃣ Boxplots of LIMIT_BAL and AGE**


<img width="1200" height="600" alt="LIM_BALvsAGE" src="https://github.com/user-attachments/assets/37edce6f-7135-4625-975c-e05ea4f5a983" />

**Insight:**

LIMIT_BAL has a wide range and many high outliers (very high credit limits).

AGE distribution is stable with fewer extreme outliers.

Younger customers seem to have lower limits; older customers show slightly higher variability.

**Business Interpretation:**
Credit limits vary a lot between customers → an important factor for credit risk prediction.


**6️⃣ Boxplots of Payment Amounts (Last 6 Months)**


<img width="1200" height="600" alt="Payment Amounts (Last 6 Months)" src="https://github.com/user-attachments/assets/7e7fd19b-64c6-45d1-b6e3-088c55e7cf42" />

**Insight:**

Similar distribution across all months.

Very high outliers indicate some customers pay unusually large amounts.

Most payments are in the lower range → many customers pay small or minimum amounts.

**Default Risk Insight:**
Consistently low payments relative to bill amounts could lead to higher default probability.

**7️⃣ ROC Curve for Logistic Regression**


<img width="600" height="500" alt="ROC Curve" src="https://github.com/user-attachments/assets/382db225-46a9-48ad-afb8-611cb04821b2" />

**Insight:**

ROC Curve is barely above the diagonal → AUC = 0.602, meaning:

Model is only slightly better than random guessing.

It does not separate default vs non-default customers effectively.

**Reason:**
Imbalanced data → logistic regression unable to learn minority class patterns.



---

### **5️⃣ Data Preprocessing**

- Label Encoding / One-Hot Encoding  
- Standardizing numerical features using **StandardScaler**  
- Train-test split for stable evaluation  

---

### **6️⃣ Model Building**

Tested multiple machine learning models:

- Logistic Regression   
- Random Forest Classifier  

---

### **7️⃣ Model Evaluation**

Evaluated using:

- Accuracy Score  
- Precision, Recall, F1-Score  
- Confusion Matrix  
- ROC-AUC Score  
- Feature Importance (for Random Forest)  

---


## 🏆 Results

### 🔹 Logistic Regression
The Logistic Regression model struggled with identifying defaulters (Class 1).

**Classification Report:**

- Precision (Class 1): **0.00**
- Recall (Class 1): **0.00**
- F1-Score (Class 1): **0.00**
- Accuracy: **0.78**
- ROC-AUC: **0.602**

➡️ The model predicts almost all samples as non-defaulters, making it ineffective for real-world credit risk applications.

---

### 🔹 Random Forest Classifier (Best Model)
The Random Forest model performed noticeably better, identifying some defaulters and improving overall recall and ROC-AUC.

**Classification Report:**

- Precision (Class 1): **0.53**
- Recall (Class 1): **0.09**
- F1-Score (Class 1): **0.16**
- Accuracy: **0.78**
- ROC-AUC: **0.675**

➡️ Although improvement is needed, Random Forest clearly outperforms Logistic Regression for detecting high-risk applicants (defaulters).

---

### 🏅 **Best Performing Model: Random Forest Classifier**

- **Accuracy:** 78%  
- **ROC-AUC:** 0.675  
- **F1-Score for Defaulters:** 0.16  
- Better at identifying high-risk customers compared to Logistic Regression  
- Recommended for further tuning and deployment  


## 🧪 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/KRISHexe520/Credit-Risk-Classification-Model.git

# Navigate to folder
cd Credit-Risk-Classification-Model

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook "Credit Risk Classification Model.ipynb"
```


🙌 Author
Krish Pandya

