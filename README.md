# **🌍 ESG Risk Analysis of S&P 500 Companies**

## **📊 Data Cleaning, Imputation, Feature Engineering & Modeling**

## **📌 Table of Contents**
1. [Introduction](#introduction)  
2. [Data Preprocessing](#data-preprocessing)  
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
4. [Feature Engineering](#feature-engineering)  
5. [Modeling & Evaluation](#modeling-evaluation)  
6. [Results & Business Insights](#results-business-insights)  
7. [Next Steps](#next-steps)  

---

## **1️⃣ Introduction**

### **🌱 What is ESG Risk & Why Does it Matter?**
**ESG (Environmental, Social, and Governance) risk** assesses how companies manage sustainability and governance issues. Investors, policymakers, and analysts use **ESG scores** to evaluate corporate responsibility and risk exposure.

This project aims to analyze the **ESG risk scores of S&P 500 companies** to uncover trends and predict risk levels using machine learning models.

### **📂 About the Dataset**
The dataset contains ESG risk ratings for **S&P 500 companies**, including:
✅ **Company Information** – Symbol, Name, Sector, Industry  
✅ **ESG Risk Scores** – Total ESG Risk Score, Environmental Risk Score, Governance Risk Score  
✅ **Other Attributes** – Full-time employees, company description  

**📌 Goal:**
- Understand **ESG risk patterns** across sectors.
- Build **machine learning models** to predict **Total ESG Risk Score**.

---

## **2️⃣ Data Preprocessing**

### **🔍 Handling Missing Values & Data Cleaning**
Common data issues:
✔ **Missing Values** – Some companies might not report all ESG factors.  
✔ **Outliers** – Extreme ESG scores may distort models.  
✔ **Irrelevant Data** – Columns like "Address" and "Description" are removed.  

**🛠 Solution:**
- Used **KNN Imputer** to handle missing values.
- Dropped duplicates.

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df[['Total ESG Risk score', 'Environment Risk Score', 'Governance Risk Score']])
df[['Total ESG Risk score', 'Environment Risk Score', 'Governance Risk Score']] = df_imputed
```

---

## **3️⃣ Exploratory Data Analysis (EDA)**

### **📊 ESG Scores Across Sectors**
**Key Findings:**
📌 **Energy & Industrial companies** have **higher ESG risk scores** due to environmental concerns.  
📌 **Tech & Finance sectors** tend to have **lower ESG risk** due to better governance policies.  
📌 **Governance risk** varies significantly across industries.  

**📌 ESG Risk Across Sectors (Visualization)**
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
sns.boxplot(x='Sector', y='Total ESG Risk score', data=df, palette='coolwarm')
plt.xticks(rotation=90)
plt.title("ESG Risk Scores Across Sectors")
plt.show()
```

---

## **4️⃣ Feature Engineering**

### **🛠 Creating Meaningful Predictors**

✔ Created **Sector ESG Risk Averages** – Comparing each company's score with the industry average.  
✔ Normalized numerical columns for **better model performance**.  
✔ Encoded categorical variables (**Sector, Industry**) using **One-Hot Encoding**.  

```python
df['Sector_Avg_Risk'] = df.groupby('Sector')['Total ESG Risk score'].transform('mean')
df['Sector_Risk_Diff'] = df['Total ESG Risk score'] - df['Sector_Avg_Risk']
```

---

## **5️⃣ Modeling & Evaluation**

### **⚡ Predicting ESG Risk Scores with Machine Learning**
Models used:
✔ **Random Forest** 🌲  
✔ **XGBoost** 🚀  
✔ **LightGBM** ⚡  

### **✅ Model Results (Mean Absolute Error - MAE)**
| Model | MAE (Lower is Better) |
|---------------|----------------------|
| **Random Forest**  | **2.04** ✅ (Best) |
| **XGBoost**      | **3.0161** ❌ (Worse) |
| **LightGBM**     | **2.8178** ❌ (Slightly Worse) |

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"Random Forest MAE: {mae_rf:.4f}")
```

---

## **6️⃣ Results & Business Insights**

### **📌 Key Takeaways**
1️⃣ **Random Forest performed the best (MAE: 2.04)** – making it the most reliable model for ESG risk prediction.  
2️⃣ **XGBoost and LightGBM underperformed** – likely due to dataset structure favoring tree-based models.  
3️⃣ **Sector plays a critical role in ESG risk** – Companies in **Energy & Basic Materials** have the highest risk.  

### **📈 Business Recommendations**
✔ **Investors** should be cautious about companies in high-risk sectors.  
✔ **Companies** can improve ESG scores by enhancing governance & environmental policies.  
✔ **Future Work** – Optimize models using **hyperparameter tuning** & try **ensemble methods**.  

