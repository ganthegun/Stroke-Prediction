# Stroke Prediction App

This project is a **Stroke Prediction App** built using Streamlit. The app allows users to input various health-related parameters and predicts the likelihood of a stroke using a pre-trained machine learning model.

## Features

- User-friendly interface for inputting health data.
- Predicts the likelihood of a stroke based on user inputs.
- Displays prediction probabilities for better understanding.
- Scales input features to match the model's training data.
- Includes a Jupyter Notebook for model training and evaluation.

## Live Demo

Try the app online: [https://ganthegun-stroke-prediction.streamlit.app/](https://ganthegun-stroke-prediction.streamlit.app/)

## Dataset

The app uses the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle. The dataset contains health-related parameters such as age, glucose level, BMI, and more.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ganthegun/Stroke-Prediction.git
    cd Stroke-Prediction
    ```

2. Set up a virtual environment:
    ```bash
    python -m venv .env
    .env\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```

2. Open the app in your browser at [http://localhost:8501](http://localhost:8501).

3. Input the required health parameters in the sidebar and view the prediction results.

## File Structure

- `streamlit_app.py`: Main application file.
- `healthcare-dataset-stroke-data.csv`: Dataset used for scaling input features.
- `model.pkl`: Pre-trained machine learning model for stroke prediction.
- `requirements.txt`: List of dependencies.
- `training.ipynb`: Jupyter Notebook for model training and evaluation.

## Dependencies

The project uses the following Python libraries:

- Streamlit
- Scikit-learn
- NumPy
- Pandas
- Jupyter (for running `training.ipynb`)

## License

This project is