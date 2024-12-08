import pandas as pd

# Example usage
x_axis_left_features = [
    'left_xtl', 'left_xbr', 'left_thumb_x_1', 'left_thumb_x_2', 
    'left_thumb_x_3', 'left_thumb_x_4', 'left_thumb_x_5', 
    'left_index_finger_x_1', 'left_index_finger_x_2', 'left_index_finger_x_3', 
    'left_index_finger_x_4', 'left_middle_finger_x_1', 'left_middle_finger_x_2', 
    'left_middle_finger_x_3', 'left_middle_finger_x_4', 'left_ring_finger_x_1', 
    'left_ring_finger_x_2', 'left_ring_finger_x_3', 'left_ring_finger_x_4', 
    'left_pinkie_finger_x_1', 'left_pinkie_finger_x_2', 'left_pinkie_finger_x_3', 
    'left_pinkie_finger_x_4'
]

x_axis_right_features = [
    'right_xtl', 'right_xbr', 'right_thumb_x_1', 'right_thumb_x_2', 
    'right_thumb_x_3', 'right_thumb_x_4', 'right_thumb_x_5', 
    'right_index_finger_x_1', 'right_index_finger_x_2', 'right_index_finger_x_3', 
    'right_index_finger_x_4', 'right_middle_finger_x_1', 'right_middle_finger_x_2', 
    'right_middle_finger_x_3', 'right_middle_finger_x_4', 'right_ring_finger_x_1', 
    'right_ring_finger_x_2', 'right_ring_finger_x_3', 'right_ring_finger_x_4', 
    'right_pinkie_finger_x_1', 'right_pinkie_finger_x_2', 'right_pinkie_finger_x_3', 
    'right_pinkie_finger_x_4'
]

y_axis_left_features = [
    'left_ytl', 'left_ybr', 'left_thumb_y_1', 'left_thumb_y_2', 
    'left_thumb_y_3', 'left_thumb_y_4', 'left_thumb_y_5', 
    'left_index_finger_y_1', 'left_index_finger_y_2', 'left_index_finger_y_3', 
    'left_index_finger_y_4', 'left_middle_finger_y_1', 'left_middle_finger_y_2', 
    'left_middle_finger_y_3', 'left_middle_finger_y_4', 'left_ring_finger_y_1', 
    'left_ring_finger_y_2', 'left_ring_finger_y_3', 'left_ring_finger_y_4', 
    'left_pinkie_finger_y_1', 'left_pinkie_finger_y_2', 'left_pinkie_finger_y_3', 
    'left_pinkie_finger_y_4'
]

y_axis_right_features = [
    'right_ytl', 'right_ybr', 'right_thumb_y_1', 'right_thumb_y_2', 
    'right_thumb_y_3', 'right_thumb_y_4', 'right_thumb_y_5', 
    'right_index_finger_y_1', 'right_index_finger_y_2', 'right_index_finger_y_3', 
    'right_index_finger_y_4', 'right_middle_finger_y_1', 'right_middle_finger_y_2', 
    'right_middle_finger_y_3', 'right_middle_finger_y_4', 'right_ring_finger_y_1', 
    'right_ring_finger_y_2', 'right_ring_finger_y_3', 'right_ring_finger_y_4', 
    'right_pinkie_finger_y_1', 'right_pinkie_finger_y_2', 'right_pinkie_finger_y_3', 
    'right_pinkie_finger_y_4'
]


def process_csv(input_file, output_file):
    global x_axis_left_features, x_axis_right_features, y_axis_left_features, y_axis_right_features
    # Step 1: Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Combine all features into a dictionary for streamlined processing
    feature_groups = {
        "x_left": x_axis_left_features,
        "x_right": x_axis_right_features,
        "y_left": y_axis_left_features,
        "y_right": y_axis_right_features
    }

    # Step 2: Process each row and append min values
    for index, row in df.iterrows():
        min_values = {}  # Store min values for the current row
        
        # Calculate the minimum for each group and adjust the features
        for group_name, features in feature_groups.items():
            # Convert features to numeric, invalid parsing will be set as NaN
            values = [pd.to_numeric(row[feature], errors='coerce') for feature in features]
            
            # Calculate the minimum value ignoring NaN
            min_value = min([value for value in values if pd.notna(value)], default=0)

            min_values[f"{group_name}_min"] = min_value  # Save the min value
            
            # Subtract the min value from each feature in the group, preserving NaN
            for feature in features:
                original_value = pd.to_numeric(row[feature], errors='coerce')
                if pd.notna(original_value):
                    df.at[index, feature] = original_value - min_value
        
        # # Append the min values to the DataFrame
        # for min_name, min_value in min_values.items():
        #     df.at[index, min_name] = min_value  # Add min value as a new column for each row

    # Step 3: Save the processed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")


# input_file = '/home/user/VisionRD_COMP/csv_folder/S1_Cheese_C1.csv'
# output_file = '/home/user/VisionRD_COMP/AI-Hackathon24/Results/output.csv'

# process_csv(input_file, output_file)