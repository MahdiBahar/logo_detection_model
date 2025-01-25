import os
import tensorflow as tf
from image_similarity_finetuned import logo_similarity_make_decision
model = tf.keras.models.load_model("/home/mahdi/logo_detection_model/data-augmentation_v4/trained_model/mobilenet_finetuned_augv4_zo_gr_ro_2l_v19.h5")


def process_images_in_folders(base_dir, model):
    """
    Function to iterate through folders, process images, and get predictions.

    Args:
        base_dir (str): Path to the base directory containing image folders.
        model: Pre-trained Keras model.

    Returns:
        list: A list of dictionaries containing the file name, folder name, and predictions.
    """
    results = []  # To store prediction results
    
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Skip non-folder files

        print(f"Processing folder: {folder_name}")
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if not (file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.svg'))):
                continue  # Skip non-image files
            
            print(f"Processing file: {file_name}")
            # Call the prediction function
            result, flag, confidence = logo_similarity_make_decision(file_path, model)
            
            # Store the results
            results.append({
                "folder": folder_name,
                "file_name": file_name,
                "result": result,
                "flag": flag,
                "confidence": confidence
            })

    return results


# Example usage
base_dir = "/home/mahdi/logo_detection_model/Test2_temp/8219/"  # Update this to your folder path
results = process_images_in_folders(base_dir, model)

# Print results
for r in results:
    print(f"Folder: {r['folder']}, File: {r['file_name']}, Result: {r['result']}, Confidence: {r['confidence']}")

import csv

with open("prediction_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["folder", "file_name", "result", "flag", "confidence"])
    writer.writeheader()
    writer.writerows(results)




# test_image = "/home/mahdi/logo_detection_model/Test2_temp/8219/8219_1.jpg"
# print(logo_similarity_make_decision(test_image, model))
