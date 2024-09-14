import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

# Initialize PaddleOCR (you only need to do this once)
ocr = PaddleOCR(use_angle_cls=True, lang='en') 

# Path to your image folder and the destination for the results
img_folder = r'E:\Career\hackathon\Amazon-ML-Challenge\small_sample'
output_folder = r'E:\Career\hackathon\Amazon-ML-Challenge\paddle_ocr_output'
os.makedirs(output_folder, exist_ok=True)

# List to store OCR results for the CSV
ocr_data = []

# Iterate over all images in the folder
for img_file in os.listdir(img_folder):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        img_path = os.path.join(img_folder, img_file)
        result = ocr.ocr(img_path, cls=True)

        # Open the image
        image = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(image)

        # Extract filename without extension for saving purposes
        file_name = os.path.splitext(img_file)[0]

        # Iterate over the OCR result and draw boxes with text
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                # Get bounding box, text, and score
                box = line[0]
                text = line[1][0]
                score = line[1][1]

                # Flatten the box coordinates into a single list
                box = [tuple(coord) for coord in box]

                # Draw the bounding box (use line for drawing the polygon)
                draw.line(box + [box[0]], fill='red', width=2)  # Connect the last point to the first

                # Draw the text along with the confidence score
                draw.text((box[0][0], box[0][1] - 10), f'{text} ({score:.2f})', fill='red')

                # Save OCR results in the list
                ocr_data.append({
                    'filename': file_name,
                    'text': text,
                    'score': score,
                    'box': box
                })

        # Save the image with boxes and labels
        output_image_path = os.path.join(output_folder, f'{file_name}_paddleocr.jpg')
        image.save(output_image_path)

# Save the OCR results to a CSV file
df = pd.DataFrame(ocr_data)
output_csv_path = os.path.join(output_folder, 'ocr_results.csv')
df.to_csv(output_csv_path, index=False)

print(f"OCR results saved to {output_csv_path}")
