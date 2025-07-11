# Car Price Prediction Model

This repository contains a machine learning project to predict car prices using regression models. The goal is to build and evaluate models that can accurately estimate a car's price based on its features like make, model, mileage, and engine size.

---

## 1. Overview

This project uses a machine learning approach to predict car prices. It includes:

* **Data preprocessing** to clean and prepare the dataset.
* **Training** two regression models: **Linear Regression** and **Random Forest Regressor**.
* **Evaluation** of the models using metrics like **R² Score** and **Mean Squared Error (MSE)**.
* **Data visualizations** to understand the dataset and model performance.

---

## 2. Features

### Data Preprocessing
* Handles **missing data**.
* Encodes **categorical variables**.
* Scales **numerical features**.

### Machine Learning Models
* **Linear Regression:** A simple, foundational model for predicting continuous values.
* **Random Forest Regressor:** An advanced ensemble method that typically offers higher accuracy.

### Model Evaluation
* **R² Score:** A key metric that shows how well the model's predictions align with actual values.
* **Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual prices.

---

## 3. Getting Started

### Requirements
To run this project, you'll need the following Python libraries:
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`

Alternatively, if you have a requirements.txt file, you can use:

Bash

pip install -r requirements.txt
Installation
Clone the repository to your local machine:

Bash

git clone [https://github.com/Nikita0728/used_car_sp_prediction_model.git)
cd prediction-model
Install the required dependencies as mentioned above.

4. Usage
Dataset
The project uses a CSV file named car.csv. Please ensure this file is present in the project directory. The dataset contains various car-related features, including the target variable, price.

Running the Jupyter Notebook
The core of the project is the Jupyter Notebook prediction_model.ipynb. To execute the code, follow these steps:

Launch Jupyter Notebook from your terminal:

Bash

jupyter notebook prediction_model.ipynb
Follow the instructions within the notebook, which will guide you through data loading, preprocessing, model training, and evaluation.

5. Model Performance
The following table summarizes the evaluation metrics for the trained models:

Model

RMSE

R² Score

Linear Regression

307564.98

0.5687

Random Forest Regressor

129600.24

0.9234


Export to Sheets
The Random Forest Regressor demonstrates superior performance with a significantly lower RMSE and a much higher R² Score, indicating a better fit to the data.

6. Data Visualization
The notebook includes several visualizations to help you understand the data and model performance.

Feature Correlation Heatmap
A heatmap is used to visualize the correlation between different features and the selling price, helping to identify the most influential factors.
<img width="599" height="491" alt="image" src="https://github.com/user-attachments/assets/d130dcb8-c9f1-4185-8875-f30b1dbc93f8" />

Scatter Plot
A scatter plot visualizes the relationship between max_power and selling_price.
<img width="567" height="448" alt="image" src="https://github.com/user-attachments/assets/3fc78f92-2965-4b2c-93d2-f2d038649951" />

7. Contributing
Contributions are welcome! If you find any issues, bugs, or have suggestions for improvements, feel free to:

Fork the repository.

Create a new feature branch (git checkout -b feature/your-feature).

Commit your changes (git commit -m 'Add new feature').

Push to the branch (git push origin feature/your-feature).

Submit a pull request.









