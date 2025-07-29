# CodeX Beverage Price Prediction

A Streamlit-based application that predicts beverage price ranges based on user inputs such as age group, income level, health concerns, purchasing habits, and more.

## Features
- Predicts a beverage's price range using a pre-trained machine learning model.
- User-friendly interface built with Streamlit.
- Flexible input options capturing various demographic and behavioral features.

## Screenshot
Below is a screenshot of the project's interface:

![Beverage Price Predictor Screenshot](Screenshot%202025-07-29%20190014.png)

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repository-name>.git
   ```
2. Navigate to the project directory:
   ```bash
   cd project-codex
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Key Project Files
- `app.py`: Main Streamlit application file.
- `final_beverage_model.pkl`: Pretrained machine learning model.
- `expected_columns.pkl`: Preprocessed feature columns used during training.
- `label_encoder_y.pkl`: Label encoder for decoding model outputs.
- `Transformed_survey_results.csv`: Processed survey data used for feature extraction.
- Jupyter Notebooks:
  - `data_cleaning_codex.ipynb`: Data cleaning steps.
  - `feature_extraction.ipynb`: Feature extraction logic.
  - `model_codex.ipynb`: Model training process.

## Dependencies
The project relies on the following Python packages (specified in `requirements.txt`):
- Streamlit
- Pandas
- NumPy
- Joblib
- scikit-learn

## License
This project is licensed under the MIT License.

## Acknowledgements
Special thanks to the contributors and users who made this project possible. 🙌