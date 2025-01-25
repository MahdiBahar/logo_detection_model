#import libraries
import numpy as np
from PIL import Image , UnidentifiedImageError
import cairosvg
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the fine-tuned model
model = tf.keras.models.load_model("/home/mahdi/logo_detection_model/data-augmentation_v4/trained_model/mobilenet_finetuned_augv4_zo_gr_ro_2l_v19.h5")


# convert svg to png
def check_and_convert_svgtopng(image_path):
    try:
        if image_path.lower().endswith(".svg"):
            print(f"Converting SVG to PNG: {image_path}")
            # Convert SVG to PNG in memory
            png_path = image_path.replace(".svg", ".png")
            cairosvg.svg2png(url=image_path, write_to=png_path)
            image_path = png_path  # Update path to use the converted PNG
        else:
            pass

        return image_path  
    
    except UnidentifiedImageError:
        print(f"Invalid image file: {image_path}")
    except Exception as e:
        print(f"Unexpected error processing {image_path}: {e}")
    return None

# # Preprocessing images
# def preprocess_image(image_path, target_size=(224, 224), remove_metadata=False):

#     try:
#         # Open image using Pillow
#         with Image.open(image_path) as img:
            
#              #Check if the image has a palette and transparency
#             if img.mode == "P" and "transparency" in img.info:
#                 # print("Converting palette-based image to RGBA.")
#                 img = img.convert("RGBA")
#             # Optionally remove metadata (fix for libpng warning)
#             elif remove_metadata and img.format == "PNG":
#                 # Remove PNG metadata by re-encoding
#                 img = img.convert("RGB")  
            
            
#             # Convert to RGB format (handles grayscale and RGBA)
#             img = img.convert("RGB")
#             # Resize the image
#             img = img.resize(target_size, Image.Resampling.LANCZOS)
#             # Convert to NumPy array and normalize pixel values
#             img_array = np.array(img) / 255.0
#             img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
#             return img_array
#     except FileNotFoundError:
#         print(f"File not found: {image_path}")
#     except UnidentifiedImageError:
#         print(f"Invalid image file: {image_path}")
#     except Exception as e:
#         print(f"Unexpected error processing {image_path}: {e}")
#     return None


def logo_similarity_make_decision (img_path, model):

    # Preprocess both images (handle all formats, including PNG fixes)

    try: 
        img_path = check_and_convert_svgtopng(img_path)
        if img_path is None:
            return ["An error is occured" , 0 , "None"]
        else: 
            # img = preprocess_image(img_path, target_size=(224, 224), remove_metadata=True)
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0  
            img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
            #check the image is invalid or not
            if img is None:
                # print(f"Skipping comparison due to invalid images: {img1_path}, {img2_path}")
                print(f"Skipping comparison due to invalid images")
                return "Invalid image(s)", 0 , "None"
            # img_array = tf.expand_dims(img, axis=0)  # Add batch dimension
            # Predict
            prediction = model.predict(img_array)
            if prediction[0][0] > 0.5:
                return "Bank Mellat Logo", 1, f'{prediction[0][0]:.4f}'
            else:
                return "Not Logo", 0 , f'{prediction[0][0]:.4f}'
    except Exception as e:
        print (e)
        return "Invalid image(s)", 0 , "None"

## Test
# image_path = "/home/mahdi/logo_detection_model/Test2_temp/8219/8219_5.jpg"
# image_path = "/home/mahdi/Phishing_Project/images/2611/2611_4.svg"
# image_path = "/home/mahdi/Phishing_Project/images/2611/2611_72.jpg"
# image_path = "/home/mahdi/logo_detection_model/Test2_temp/8219/8219_5.jpg"
# image_path = "/home/mahdi/Datasets/create_new_test_dataset/new_false_tag_logo/1701/1701_1.jpg"
# image_path = "/home/mahdi/logo_detection_model/Test3_added/1/1215_1.jpg"
# image_path = "/home/mahdi/logo_detection_model/Test3_added/1/571_1.jpg"
image_path = "/home/mahdi/logo_detection_model/Test3_added/1/3171_2.jpg"

result, flag, confidence = logo_similarity_make_decision(image_path, model)
print(f"Result: {result}, with flag : {flag} and Confidence: {confidence}")
