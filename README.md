# Car Price Prediction Model

This repository contains a machine learning project to predict car prices using regression models. The goal is to build and evaluate models that can accurately estimate a car's price based on its features like make, model, mileage, and engine size.

-----

## 1\. Overview

This project uses a machine learning approach to predict car prices. It includes:

  * **Data preprocessing** to clean and prepare the dataset.
  * **Training** two regression models: **Linear Regression** and **Random Forest Regressor**.
  * **Evaluation** of the models using metrics like **R² Score** and **Mean Squared Error (MSE)**.
  * **Data visualizations** to understand the dataset and model performance.

-----

## 2\. Features

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

-----

## 3\. Getting Started

### Requirements

To run this project, you'll need the following Python libraries:

  * `pandas`
  * `numpy`
  * `matplotlib`
  * `seaborn`
  * `scikit-learn`

You can install them with `pip`. If you have a `requirements.txt` file, run:

```bash
pip install -r requirements.txt
```

Otherwise, you can install them manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Installation

1.  Clone the repository to your local machine:

<!-- end list -->

```bash
git clone https://github.com/your-username/prediction-model.git
cd prediction-model
```

2.  Install the required dependencies as mentioned above.

-----

## 4\. Usage

### Dataset

This project uses a dataset from a CSV file named `car.csv`. Make sure this file is in your project directory before you begin. It should contain various car attributes, including the target variable, **price**.

### Running the Notebook

The main analysis is in the Jupyter Notebook `prediction_model.ipynb`. To run it, follow these steps:

1.  Launch Jupyter Notebook from your terminal:

<!-- end list -->

```bash
jupyter notebook prediction_model.ipynb
```

2.  Open the notebook and run the cells in order. The notebook guides you through each step of the process, including data loading, preprocessing, model training, and evaluation.

-----

## 5\. Model Training & Evaluation

The notebook performs the following steps:

1.  **Data Loading:** The `car.csv` file is loaded into a pandas DataFrame.
2.  **Preprocessing:** The data is cleaned and prepared for modeling.
3.  **Model Training:** **Linear Regression** and **Random Forest Regressor** models are trained on the preprocessed data.
4.  **Evaluation:** The performance of each model is measured.

### Model Performance

| Model | RMSE | R² Score |
| :--- | :--- | :--- |
| **Linear Regression** | 307564.98 | 0.5687 |
| **Random Forest Regressor** | 129600.24 | 0.9234 |

The **Random Forest Regressor** model significantly outperforms the **Linear Regression** model, demonstrating its ability to capture more complex relationships within the data.

-----

## 6\. Data Visualization

The notebook includes several visualizations to help you understand the data and model performance.

### Feature Correlation Heatmap

This heatmap shows the correlation between different features and the selling price, helping to identify which factors are most influential.
<img width="599" height="491" alt="image" src="https://github.com/user-attachments/assets/b221d183-6e75-4f6f-8d96-f87793fc7631" />

### Scatter Plot

This scatter plot visualizes the relationship between `max_power` and `selling_price`, providing a clear view of how these two variables are related.
<img width="567" height="448" alt="image" src="https://github.com/user-attachments/assets/01f38ffd-e0c0-44b3-860b-b42ce30cf1cd" />

-----

## 7\. Contributing

We welcome contributions to this repository\! If you find a bug, have an idea for a new feature, or want to improve the code, please feel free to:

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/your-feature`).
3.  Commit your changes (`git commit -m 'Add new feature'`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Open a **pull request**.

