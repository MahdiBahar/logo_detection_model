import os
import tensorflow as tf
from image_similarity_finetuned import logo_similarity_make_decision
import csv
import shutil

# model = tf.keras.models.load_model("/home/mahdi/logo_detection_model/data-augmentation_v4/trained_model/mobilenet_finetuned_augv4_zo_gr_ro_2l_v19.h5")
model = tf.keras.models.load_model("//home/mahdi/logo_detection_model/data-augmentation_v2/trained_model/mobilenet_finetuned_add_data_3aug_2trainlayer_v14.h5")

def process_images_in_folders(base_dir, model):
   
    save_dir = "/home/mahdi/logo_detection_model/flagged_images_thr0.8/" 
    os.makedirs(save_dir, exist_ok=True)
    results = []  # To store prediction results

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            print(f"Skipping non-folder: {folder_name}")
            continue  # Skip non-folder files

        print(f"Processing folder: {folder_name}")
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Check if the file is an image
            if not file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.svg')):
                print(f"Skipping non-image file: {file_name}")
                continue

            print(f"Processing file: {file_name}")
            # Call the prediction function
            result, flag, confidence = logo_similarity_make_decision(file_path, model)

            # Check if the result is valid
            if result == "Invalid image(s)":
                print(f"Invalid image skipped: {file_name}")
                continue

            # Store the results
            results.append({
                "folder": folder_name,
                "file_name": file_name,
                "result": result,
                "flag": flag,
                "confidence": confidence
            })
            # Save flagged images
            if flag == 1:  # If the image has a flag of 1
                # Create a subfolder in the save directory for the corresponding class
                class_save_dir = os.path.join(save_dir, folder_name)
                os.makedirs(class_save_dir, exist_ok=True)
                shutil.copy(file_path, os.path.join(class_save_dir, file_name))


    return results

# Example usage
# base_dir = "/home/mahdi/Datasets/new_false_tag_logo/"  # Update this to your folder path
base_dir = "/home/mahdi/Datasets/50/"  # Update this to your folder path

results = process_images_in_folders(base_dir, model)

# Print results
if results:
    for r in results:
        print(f"Folder: {r['folder']}, File: {r['file_name']}, Result: {r['result']}, Confidence: {r['confidence']}")
else:
    print("No results were generated. Check if images are valid and present in the folder.")



if results:
    with open("prediction_results_thr0.8.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "file_name", "result", "flag", "confidence"])
        writer.writeheader()
        writer.writerows(results)
else:
    print("No data to write to CSV. The `results` list is empty.")



# test_image = "/home/mahdi/logo_detection_model/Test2_temp/8219/8219_1.jpg"
# print(logo_similarity_make_decision(test_image, model))
