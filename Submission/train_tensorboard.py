import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    f1_score, 
    log_loss
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib

# Added for TensorBoard logging
from torch.utils.tensorboard import SummaryWriter

# Additional imports for mAP calculation
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='runs/exp1')

# -----------------------------------
# Load Training and Testing Datasets
# -----------------------------------
train_df = pd.read_csv('transformed3_s1_non_output.csv')
test_df = pd.read_csv('transformed3_s1_output.csv')

# Shuffle the data
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Data shuffled successfully!")

# Replace string "Nan" with actual NaN values and ensure numeric columns
def clean_data(df, columns_to_clean):
    """Replaces 'Nan' with NaN and converts specified columns to numeric."""
    df.replace('Nan', np.nan, inplace=True)
    for col in columns_to_clean:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Columns to clean (all columns except 'action', 'frame_id', 'frame_name')
columns_to_clean = [
    col for col in train_df.columns if col not in ['action', 'frame_id', 'frame_name']
]

# Clean train and test datasets
train_df = clean_data(train_df, columns_to_clean)
test_df = clean_data(test_df, columns_to_clean)

# # Add derived features for both datasets
for df in [train_df, test_df]:
    df['x_diff'] = df['left_xtl'] - df['right_xtl']
    df['y_diff'] = df['left_ytl'] - df['right_ytl']
    df['bbox_area'] = (df['left_xbr'] - df['left_xtl']) * (df['left_ybr'] - df['left_ytl'])
    df['aspect_ratio'] = (df['left_xbr'] - df['left_xtl']) / (df['left_ybr'] - df['left_ytl'] + 1e-6)
    df['diagonal_length'] = np.sqrt(df['x_diff']**2 + df['y_diff']**2)
    df['bbox_perimeter'] = 2 * ((df['left_xbr'] - df['left_xtl']) + (df['left_ybr'] - df['left_ytl']))

# Split features and targets for training and testing data
X_train = train_df.drop(columns=train_df.columns[:11])
y_train = train_df['action']

X_test = test_df.drop(columns=train_df.columns[:11])
y_test = test_df['action']

# -----------------------------------
# Compute Class Weights for Primary Model
# -----------------------------------
class_weights = compute_class_weight('balanced', classes=y_train.unique(), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(y_train.unique(), class_weights)}

# Adjust the class weight for 'take' slightly
if 'take' in class_weight_dict:
    class_weight_dict['take'] *= 0.8  # Reduce emphasis on 'take'

print(f"Class weights: {class_weight_dict}")

# -----------------------------------
# Define and Train the Primary Model
# -----------------------------------
primary_pipeline = make_pipeline(
    SimpleImputer(strategy='constant', fill_value=0.0),
    StandardScaler(),
    RandomForestClassifier(class_weight=class_weight_dict, random_state=1235)
)

primary_pipeline.fit(X_train, y_train)

# Save the primary model
primary_model_filename = 'primary_random_forest_model.pkl'
joblib.dump(primary_pipeline, primary_model_filename)
print(f"Primary model saved to {primary_model_filename}")

# Predict with the primary model
primary_predictions = primary_pipeline.predict(X_test)
primary_probabilities = primary_pipeline.predict_proba(X_test)

# Extract classes from the primary model
classes = primary_pipeline.named_steps['randomforestclassifier'].classes_

# -----------------------------------
# Compute Training Accuracy (Existing Metric)
# -----------------------------------
train_predictions = primary_pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Training Accuracy: {train_accuracy:.4f}")
writer.add_scalar('Accuracy/Train', train_accuracy, 0)

# -----------------------------------
# Compute Training Loss and Validation Loss (New Metrics)
# We'll use log_loss as a proxy for "loss".
train_loss = log_loss(y_train, primary_pipeline.predict_proba(X_train))
val_loss = log_loss(y_test, primary_probabilities)
print(f"Training Loss (log_loss): {train_loss:.4f}")
print(f"Validation Loss (log_loss): {val_loss:.4f}")
writer.add_scalar('Loss/Train', train_loss, 0)
writer.add_scalar('Loss/Val', val_loss, 0)

# -----------------------------------
# Compute mAP (mean Average Precision) on Test Set (Added Metric)
Y_test_binary = label_binarize(y_test, classes=classes)
mAP = 0.0
for i in range(len(classes)):
    mAP += average_precision_score(Y_test_binary[:, i], primary_probabilities[:, i])
mAP /= len(classes)

print(f"mAP (based on primary probabilities): {mAP:.4f}")
writer.add_scalar('mAP/Test', mAP, 0)

# -----------------------------------
# Extract and Process Ambiguous `background` Predictions
# -----------------------------------
background_indices = [
    i for i, pred in enumerate(primary_predictions) if pred == 'background'
]
background_X = X_test.iloc[background_indices]
background_y = y_test.iloc[background_indices]

# Create labels for the secondary classifier: `background` vs. `not background`
secondary_labels = [
    'background' if true == 'background' else 'not_background'
    for true in background_y
]

# -----------------------------------
# Compute Class Weights for Secondary Model
# -----------------------------------
secondary_class_weights = compute_class_weight(
    'balanced', classes=np.unique(secondary_labels), y=secondary_labels
)
secondary_class_weight_dict = {
    cls: weight for cls, weight in zip(np.unique(secondary_labels), secondary_class_weights)
}
print(f"Secondary class weights: {secondary_class_weight_dict}")

# -----------------------------------
# Define and Train the Secondary Model
# -----------------------------------
secondary_pipeline = make_pipeline(
    SimpleImputer(strategy='constant', fill_value=0.0),
    StandardScaler(),
    RandomForestClassifier(class_weight=secondary_class_weight_dict, random_state=1235)
)

secondary_pipeline.fit(background_X, secondary_labels)

# Save the secondary model
secondary_model_filename = 'secondary_random_forest_model.pkl'
joblib.dump(secondary_pipeline, secondary_model_filename)
print(f"Secondary model saved to {secondary_model_filename}")

# -----------------------------------
# Refine `background` Predictions with Confidence Threshold
# -----------------------------------
confidence_threshold_secondary = 0.3
refined_predictions = []
for i, (prob, pred) in enumerate(zip(primary_probabilities, primary_predictions)):
    if pred == 'background':
        # Prepare data in consistent format for secondary pipeline
        refined_sample = pd.DataFrame([X_test.iloc[i]])
        refined_sample = pd.DataFrame(
            secondary_pipeline.named_steps['simpleimputer'].transform(refined_sample),
            columns=X_train.columns
        )
        is_background = secondary_pipeline.predict(refined_sample)[0]
        refined_sample_proba = secondary_pipeline.predict_proba(refined_sample)[0]
        background_proba = refined_sample_proba[
            np.where(np.unique(secondary_labels) == 'background')[0][0]
        ]
        if is_background == 'not_background' or background_proba < confidence_threshold_secondary:
            # Assign the second-highest probability class
            second_highest_index = np.argsort(prob)[-2]
            refined_predictions.append(classes[second_highest_index])
        else:
            refined_predictions.append('background')
    else:
        refined_predictions.append(pred)

# -----------------------------------
# Evaluate Refined Predictions
# -----------------------------------
accuracy = accuracy_score(y_test, refined_predictions)
print(f"Accuracy after refinement: {accuracy:.4f}")

# Log accuracy after refinement to TensorBoard
writer.add_scalar('Accuracy/Refined', accuracy, 0)

# Compute F1 score (New Metric)
f1 = f1_score(y_test, refined_predictions, average='weighted')
print(f"F1 Score (Weighted) after refinement: {f1:.4f}")
writer.add_scalar('F1/Refined', f1, 0)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, refined_predictions))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, refined_predictions, labels=classes)
print("\nConfusion Matrix:")
print(conf_matrix)

# -----------------------------------
# Analyze Misclassified Samples
# -----------------------------------
misclassified = pd.DataFrame({'True': y_test, 'Predicted': refined_predictions})
misclassified = misclassified[misclassified['True'] != misclassified['Predicted']]
misclassified_counts = misclassified['True'].value_counts()

print("\nClasses with most misclassifications:")
print(misclassified_counts)

# Save misclassified samples to a CSV file
misclassified.to_csv('misclassified_samples_refined.csv', index=False)
print("\nMisclassified samples saved to 'misclassified_samples_refined.csv'")

# Close the TensorBoard writer
writer.close()
