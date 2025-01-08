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





**Note:**  
*Alan da Silva Martins*  
*Email: alansmartinss@hotmail.com*  
*LinkedIn: [linkedin.com/in/alansmartinss](https://linkedin.com/in/alansmartinss)*  