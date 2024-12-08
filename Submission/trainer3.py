import os
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

def process_predictions(input_csv_path, output_folder_path):
    """
    Processes predictions for the given input CSV file using pre-trained models and
    saves the results in a text file in the specified output folder.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_folder_path (str): Path to the output folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder_path, exist_ok=True)

    # Load the dataset
    test_df = pd.read_csv(input_csv_path)

    # Replace string "Nan" with actual NaN values and ensure numeric columns
    def clean_data(df, columns_to_clean):
        """Replaces 'Nan' with NaN and converts specified columns to numeric."""
        df.replace('Nan', np.nan, inplace=True)
        for col in columns_to_clean:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    # Columns to clean (all columns except 'frame_id', 'frame_name')
    columns_to_clean = [
        col for col in test_df.columns if col not in ['frame_id', 'frame_name']
    ]

    # Clean test dataset
    test_df = clean_data(test_df, columns_to_clean)

    # Add derived features for the dataset
    test_df['x_diff'] = test_df['left_xtl'] - test_df['right_xtl']
    test_df['y_diff'] = test_df['left_ytl'] - test_df['right_ytl']
    test_df['bbox_area'] = (test_df['left_xbr'] - test_df['left_xtl']) * (test_df['left_ybr'] - test_df['left_ytl'])
    test_df['aspect_ratio'] = (test_df['left_xbr'] - test_df['left_xtl']) / (test_df['left_ybr'] - test_df['left_ytl'] + 1e-6)
    test_df['diagonal_length'] = np.sqrt(test_df['x_diff']**2 + test_df['y_diff']**2)
    test_df['bbox_perimeter'] = 2 * ((test_df['left_xbr'] - test_df['left_xtl']) + (test_df['left_ybr'] - test_df['left_ytl']))

    # Split features for testing data (drop columns before the 11th column)
    X_test = test_df.drop(columns=test_df.columns[:10])

    # Load pre-trained models
    primary_model_filename = 'primary_random_forest_model.pkl'
    secondary_model_filename = 'secondary_random_forest_model.pkl'

    primary_pipeline = joblib.load(primary_model_filename)
    secondary_pipeline = joblib.load(secondary_model_filename)

    print("Models loaded successfully!")

    # Predict with the primary model
    primary_predictions = primary_pipeline.predict(X_test)
    primary_probabilities = primary_pipeline.predict_proba(X_test)

    # Extract classes from the primary model
    classes = primary_pipeline.named_steps['randomforestclassifier'].classes_

    # Extract and process ambiguous `background` predictions
    background_indices = [
        i for i, pred in enumerate(primary_predictions) if pred == 'background'
    ]
    background_X = X_test.iloc[background_indices]

    # Refine `background` predictions with confidence threshold
    confidence_threshold_secondary = 0.3
    refined_predictions = []
    for i, (prob, pred) in enumerate(zip(primary_probabilities, primary_predictions)):
        if pred == 'background':
            refined_sample = pd.DataFrame([X_test.iloc[i]])
            refined_sample = pd.DataFrame(
                secondary_pipeline.named_steps['simpleimputer'].transform(refined_sample),
                columns=X_test.columns
            )
            is_background = secondary_pipeline.predict(refined_sample)[0]
            refined_sample_proba = secondary_pipeline.predict_proba(refined_sample)[0]
            background_proba = refined_sample_proba[
                np.where(np.unique(['background', 'not_background']) == 'background')[0][0]
            ]
            if is_background == 'not_background' or background_proba < confidence_threshold_secondary:
                second_highest_index = np.argsort(prob)[-2]
                refined_predictions.append(classes[second_highest_index])
            else:
                refined_predictions.append('background')
        else:
            refined_predictions.append(pred)

    # Save refined predictions to a text file
    output_file_name = os.path.basename(input_csv_path).replace('.csv', '.txt')
    output_file_path = os.path.join(output_folder_path, output_file_name)
    with open(output_file_path, 'w') as f:
        for prediction in refined_predictions:
            f.write(f"{prediction}\n")

    print(f"Predictions saved to {output_file_path}")


