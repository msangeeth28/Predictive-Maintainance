# End-to-End Predictive Maintenance using Machine Learning

## Project Overview
This project aims to build an End-to-End Machine Learning solution for predictive maintenance. By analyzing machine parameters such as temperature, rotational speed, torque, and tool wear, the model can predict whether a machine is likely to fail. Additionally, if a failure is predicted, the system classifies the specific failure type to recommend actionable maintenance steps.

## Objectives
1. **Predict Machine Failure**: A binary classification model to predict whether a machine will fail.
2. **Predict Failure Type**: A multi-class classification model to predict the specific type of failure (e.g., Tool Wear, Heat Dissipation, Power, Overstrain, or Random Failure).
3. **Actionable Recommendations**: Provide maintenance recommendations based on the predicted failure type via a real-time dashboard.

## Dataset
The project uses the **AI4I 2020 Predictive Maintenance Dataset**. It contains 10,000 data points with features such as Air temperature, Process temperature, Rotational speed, Torque, and Tool wear.

## Libraries
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `joblib`
- `shap`
- `streamlit`
- `pytest`

## Installation
1. Ensure you have Python 3.8+ installed.
2. Clone or extract the project repository.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## How to Run Notebook
1. Open a terminal or command prompt in the project directory.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `Predictive_Maintenance.ipynb` and run all cells sequentially from top to bottom. This will train the models and save them in the `models/` directory.

## How to Run Streamlit
1. After running the notebook to generate the models, execute the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. A browser window will open automatically. Input the machine parameters to get predictions and maintenance recommendations.

## Results
- The binary classification model achieved high accuracy; however, due to class imbalance, failure detection performance needs further improvement.
- The best failure type classification model effectively distinguishes between different failure modes.
- Important features driving failures include mechanical power, tool wear, and temperature differences.

## Future Improvements
- Collect more data for less frequent failure types to handle class imbalance even better.
- Incorporate time-series or sequential data for real-time sensor streams.
- Deploy the Streamlit application to a cloud platform like AWS, Heroku, or Streamlit Community Cloud.

