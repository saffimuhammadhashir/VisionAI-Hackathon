import os
import numpy as np
import pandas as pd
from lxml import etree
import transform as tcsv  # Assuming this module contains the `process_csv` function
import trainer3 as trans_model
# Input directories
xml_dir = '/home/user/VisionRD_COMP/xml'  # Directory containing XML files
actions_dir = '/home/user/VisionRD_COMP/actions'  # Directory containing TXT action files
output_dir = '/home/user/VisionRD_COMP/csv_folder_result'  # Directory to save the intermediate and final CSV files
output_dir_inter = '/home/user/VisionRD_COMP/csv_folder_result_inter'  # Directory to save the intermediate and final CSV files
output_txt_folder= '/home/user/VisionRD_COMP/ResultTxtOutput'
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

import os

def calculate_accuracy(actual_file, sample_file):
    """
    Compare two text files line by line and calculate accuracy.

    Args:
        actual_file (str): Path to the actual text file.
        sample_file (str): Path to the sample text file.

    Returns:
        float: Accuracy percentage.
    """
    try:
        # Open both files and read lines
        with open(actual_file, 'r') as f1, open(sample_file, 'r') as f2:
            actual_lines = f1.readlines()
            sample_lines = f2.readlines()

        # Ensure both files have the same number of lines
        if len(actual_lines) != len(sample_lines):
            print(f"Warning: Files {os.path.basename(actual_file)} and {os.path.basename(sample_file)} have different numbers of lines.")
        
        # Compare lines and count matches
        total_lines = min(len(actual_lines), len(sample_lines))
        matches = sum(
            1 for i in range(total_lines) if actual_lines[i].strip() == sample_lines[i].strip()
        )

        # Calculate accuracy
        accuracy = (matches / total_lines) * 100 if total_lines > 0 else 0

        return accuracy, matches, total_lines

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 0, 0, 0

def process_directories(actual_dir, sample_dir):
    """
    Process all files in the actual and sample directories and compare them.

    Args:
        actual_dir (str): Path to the directory containing actual files.
        sample_dir (str): Path to the directory containing sample files.
    """
    actual_files = set(os.listdir(actual_dir))
    sample_files = set(os.listdir(sample_dir))

    # Find common files between the two directories
    common_files = actual_files.intersection(sample_files)

    if not common_files:
        print("No matching files found in the directories.")
        return

    print(f"Processing {len(common_files)} matching files...\n")

    for file_name in common_files:
        actual_file_path = os.path.join(actual_dir, file_name)
        sample_file_path = os.path.join(sample_dir, file_name)

        accuracy, matches, total_lines = calculate_accuracy(actual_file_path, sample_file_path)
        print(f"File: {file_name}")
        print(f"  Accuracy: {accuracy:.2f}% ({matches}/{total_lines} lines match)\n")



def process_files(xml_file, actions_file, intermediate_csv):
    # Parse the XML file
    tree = etree.parse(xml_file)
    root = tree.getroot()

    # Load actions from the .txt file
    with open(actions_file, 'r') as f:
        actions = [line.strip() for line in f.readlines()]

    # Dictionary to hold data by frame
    frames_data = {}
    max_points = {"left": {}, "right": {}}  # To track max points for each hand and finger

    # First Pass: Collect data and determine max points
    for image in root.findall('.//image'):
        frame_id = image.get("id")
        frame_name = image.get("name")
        
        # Extract numeric part from frame_id
        try:
            if frame_id.startswith('frame_'):
                frame_index = int(frame_id.split('_')[1]) - 1  # Convert to 0-based index
            elif frame_id.isdigit():
                frame_index = int(frame_id) - 1  # Numeric-only IDs
            else:
                raise ValueError("Unrecognized format")
        except (IndexError, ValueError):
            print(f"Warning: Unexpected frame_id format '{frame_id}' in file {xml_file}. Using default index -1.")
            frame_index = -1

        action = actions[frame_index] if 0 <= frame_index < len(actions) else "Unknown"

        if frame_id not in frames_data:
            frames_data[frame_id] = {
                "frame_id": frame_id,
                "frame_name": frame_name,
                "action": action,
                "left_hand": {},
                "right_hand": {},
            }

        # Process boxes
        for box in image.findall(".//box"):
            hand_type = box.find(".//attribute[@name='hand_type']").text
            coords = {
                "xtl": box.get("xtl"),
                "ytl": box.get("ytl"),
                "xbr": box.get("xbr"),
                "ybr": box.get("ybr")
            }
            frames_data[frame_id][f"{hand_type}_hand"].update(coords)

        # Process polylines
        for polyline in image.findall(".//polyline"):
            hand_type = polyline.find(".//attribute[@name='hand_type']").text
            label = polyline.get("label")
            points = polyline.get("points")
            points = [
                tuple(map(int, map(float, point.split(','))))
                for point in points.split(";")
            ]
            frames_data[frame_id][f"{hand_type}_hand"][label] = points

            # Track the maximum number of points dynamically
            if label not in max_points[hand_type]:
                max_points[hand_type][label] = 0
            max_points[hand_type][label] = max(max_points[hand_type][label], len(points))

    # Second Pass: Combine data into DataFrame
    frames_combined = []
    for frame_id, data in frames_data.items():
        left_hand = data["left_hand"]
        right_hand = data["right_hand"]

        combined_row = {
            # "action": data["action"],  # Add action as the first column
            "frame_id": data["frame_id"],
            "frame_name": data["frame_name"],
            "left_xtl": left_hand.get("xtl", "Nan"),
            "left_ytl": left_hand.get("ytl", "Nan"),
            "left_xbr": left_hand.get("xbr", "Nan"),
            "left_ybr": left_hand.get("ybr", "Nan"),
            "right_xtl": right_hand.get("xtl", "Nan"),
            "right_ytl": right_hand.get("ytl", "Nan"),
            "right_xbr": right_hand.get("xbr", "Nan"),
            "right_ybr": right_hand.get("ybr", "Nan"),
        }

        # Add individual polyline points for each hand and finger dynamically
        for hand_type, hand_data in [("left", left_hand), ("right", right_hand)]:
            for finger, max_point_count in max_points[hand_type].items():
                points = hand_data.get(finger, [])
                for i in range(max_point_count):
                    if i < len(points):
                        x, y = points[i]
                    else:
                        x, y = -1, -1  # Fill missing points with -1
                    combined_row[f"{hand_type}_{finger}_x_{i+1}"] = x
                    combined_row[f"{hand_type}_{finger}_y_{i+1}"] = y

        frames_combined.append(combined_row)

    # Create DataFrame and save intermediate CSV
    df_combined = pd.DataFrame(frames_combined)
    df_combined.to_csv(intermediate_csv, index=False)

# Process each file pair in the directories

for xml_filename in os.listdir(xml_dir):
    if xml_filename.endswith('.xml'):
        base_name = os.path.splitext(xml_filename)[0]
        xml_path = os.path.join(xml_dir, xml_filename)
        actions_path = os.path.join(actions_dir, f"{base_name}.txt")
        intermediate_csv_path = os.path.join(output_dir_inter, f"{base_name}_intermediate.csv")
        final_csv_path = os.path.join(output_dir, f"{base_name}.csv")

        if os.path.exists(actions_path):
            print(f"Processing: {xml_filename} with {base_name}.txt")
            process_files(xml_path, actions_path, intermediate_csv_path)
            
            # Process intermediate CSV with process_csv function
            tcsv.process_csv(intermediate_csv_path, final_csv_path)
            # model call
            trans_model.process_predictions(final_csv_path,output_txt_folder)

        else:
            print(f"Warning: Actions file {base_name}.txt not found for {xml_filename}")

process_directories(actions_dir, output_txt_folder)