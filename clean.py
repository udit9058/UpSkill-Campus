import os
import cv2

def clean_images(dataset_path, output_path):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate through each file in the dataset folder
    for file_name in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file_name)

        # Check if the file is an image
        if is_image(file_path):
            try:
                # Read the image
                image = cv2.imread(file_path)

                # Perform any necessary image cleaning or preprocessing operations here
                # For example, you can resize the image, remove noise, or adjust brightness and contrast

                # Save the cleaned/preprocessed image to the output directory
                output_image_path = os.path.join(output_path, file_name)
                cv2.imwrite(output_image_path, image)

            except Exception as e:
                print(f"Error processing image: {file_name}")
                print(e)

def is_image(file_path):
    # Check if the file has a valid image extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in valid_extensions

# Specify the path to your dataset folder and the desired output path for the cleaned images
dataset_path = "D:\internships\ds\Project5_Ag_Crop and weed detection\data"
output_path = "D:\internships\ds\Project5_Ag_Crop and weed detection\output"

# Clean and preprocess the images in the dataset
clean_images(dataset_path, output_path)
