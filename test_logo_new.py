import logging
from DatabaseManager import DatabaseManager 
import os
from dotenv import load_dotenv
from image_similarity_finetuned import logo_similarity_make_decision
from pathlib import Path
import time
import shutil  # For copying files


# Load environment variables
load_dotenv()

# Set up logging
log_file = "logging_logo_detection.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Logs to terminal as well
    ]
)
logger = logging.getLogger(__name__)

# Database connection parameters
db_params = {
    'dbname': os.getenv("DB_NAME"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'host': os.getenv("DB_HOST"),
    'port': os.getenv("DB_PORT")
}

# Instantiate the DatabaseManager
db_manager = DatabaseManager(db_params)

try:
    images = db_manager.fetch_url_for_logo_detection()
except Exception as e:
    logger.error(f"Error fetching URLs from database: {e}")
    images = []



# valid_img = ['BM_LOGO-00.png', 'BM_LOGO-01.png', 'BM_LOGO-02.png', 'BM_LOGO-03.png', 'BM_LOGO-04.png','BM_LOGO-05.png', 'BM_LOGO-06.jpg']
# valid_img_path = "/home/yegane/mellat/phishing/phishing code/Valid_images/"

# Define the folder to save images with detected logos
detected_logo_folder = "/home/yegane/mellat/phishing/phishing code/detected_logos"
# Ensure the folder exists
os.makedirs(detected_logo_folder, exist_ok=True)
model = "/home/yegane/mellat/phishing/phishing code/mobilenet_finetuned_augv4_zo_gr_ro_2l_v19.h5"
for items in images:
    url_id = items["url_id"] 
    image_path = items["url_image"]
    
    try:
        logger.info(f"Processing folder path: {image_path}")

        # Check if the path exists and is a directory
        if not os.path.isdir(image_path):
            logger.warning(f"Invalid or non-existent directory for url_id {url_id}: {image_path}")
            continue

        logo_detected = False  # Default result

        # Iterate through all files in the folder
        for filename in os.listdir(image_path):
            file_path = os.path.join(image_path, filename)

            # Check if the file is a PNG image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.svg', '.bin')):
                try:
                    logger.info("********************************************************")
                    logger.info(f"Processing image: {filename}")

                    # Call make_decision on each image
                    decision_result = logo_similarity_make_decision(file_path, model)
                    
                    # Extract items from the result
                    Similarity_detaile = decision_result[0]
                    decision_flag = decision_result[1]  # Detection status
                    similarity_score = decision_result[2]
        


                    logger.info(f"Similarity Deatile : {Similarity_detaile} ")
                    logger.info(f"Decision Flag is {decision_flag}") 
                    logger.info(f"Similarity Score is : {similarity_score}")

                    if decision_flag:

                        # Save the image to the designated folder
                        destination_path = os.path.join(detected_logo_folder, filename)
                        shutil.copy(file_path, destination_path)
                        logger.info(f"Image {filename} saved to {detected_logo_folder}")
                        
                        logo_detected = True  # Update the result to 1
                        logger.info(f"Logo detected in file: {filename}. Skipping remaining files.")
                        break  # Exit the loop and move to the next URL
                
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
        
        # Save the result (0 or 1) for the current URL in the database
        try:
            db_manager.save_logo_detection(url_id, logo_detected)
            logger.info(f"Detection result for URL ID {url_id} updated to {logo_detected} in database successfully.")
        except Exception as e:
            logger.error(f"Error saving detection result for URL ID {url_id}: {e}")

    except Exception as e:
        logger.error(f"Error processing folder for URL ID {url_id}: {e}")
