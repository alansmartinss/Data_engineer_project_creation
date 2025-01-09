# My Project
This project aims to showcase the end-to-end data workflow, including data collection, cleaning, preprocessing, and adjustment. It also covers the construction and maintenance of data pipelines, the development of supervised machine learning models, and the deployment and maintenance of a fully functional ML system.


## Requirements
- Python 3.8+
- Jupyter Notebook

### **Analysis Notebook Overview**

The analysis notebook is designed to perform a complete exploratory and preprocessing workflow for the dataset used in this project. It leverages modularized functions from the `basic_analysys.py` file, located in the `created_functions` folder, ensuring reusable and maintainable code.

The notebook follows these key steps:
1. **Dataset Loading**: Reads the dataset (`heart_disease.csv`) from the `Datasets` folder.
2. **Exploratory Analysis**: Utilizes functions to inspect the dataset's structure, identify missing values, analyze numerical and categorical features, and visualize distributions and correlations.
3. **Data Cleaning**: 
   - Removes rows with more than 4 missing values to ensure data quality.
   - Imputes remaining missing values:
     - Numerical columns are filled with their mean.
     - Categorical columns are filled with their mode.
4. **Output**: Saves the cleaned dataset as `adjusted_dataset.csv` in the `Datasets` folder.

---

### Section 2: Pre-Modeling and Feature Engineering

This notebook focuses on **testing sampling strategies** and performing **feature engineering** to prepare the dataset for supervised machine learning models. Below are the key steps performed in this notebook:

---

#### **1. Sampling Techniques**
- The notebook evaluates different sampling methods to split the dataset into **training** and **testing** subsets:
  - **Simple Random Sampling:** Ensures a random and unbiased split.
  - **Stratified Sampling by Target:** Maintains the same distribution of the target variable across both subsets. This method was selected due to its ability to retain the balance of target classes, which is crucial for training robust machine learning models.

#### **2. Data Analysis and Exploration**
- Comprehensive data exploration was conducted on both training and testing subsets to ensure their quality and consistency:
  - **Proportions Check:** Verified the distribution of the target variable and other important attributes across the subsets.
  - **Visual Analysis:** Histograms and boxplots were created to compare the distributions of numerical features across training and testing datasets.

#### **3. Feature Engineering**
New features were created to enrich the dataset, enhancing its predictive power:
- **Numerical Feature Ratios:** Relationships between key numerical variables were calculated (e.g., Cholesterol-to-Triglyceride Ratio).
- **Lifestyle Scores:** Aggregated variables such as exercise habits, smoking, and alcohol consumption into a risk score.
- **Categorical Feature Interactions:** Created indicators for high-risk combinations (e.g., High Blood Pressure and Family History of Heart Disease).
- **Categorization of Numerical Features:** Variables like age and BMI were grouped into bins (e.g., Age Group and BMI Category) for better interpretability.

#### **4. Evaluation**
- The created features and dataset splits were thoroughly analyzed to ensure quality and relevance for the machine learning pipeline.

---

This notebook establishes the foundation for robust model training by ensuring that the dataset is well-prepared and engineered with domain-specific insights.




**Note:**  
*Alan da Silva Martins*  
*Email: alansmartinss@hotmail.com*  
*LinkedIn: [linkedin.com/in/alansmartinss](https://linkedin.com/in/alansmartinss)*  