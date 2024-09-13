import os
import numpy as np
import cv2
import pandas as pd

def crop(pts, image):
    """
    Takes inputs as 8 points and returns a cropped, masked image with a white background.
    """
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    cropped = image[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    bg = np.ones_like(cropped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    return dst2

def generate_words(image_name, score_bbox, image, output_dir):
    """
    Generate and save cropped word images based on bounding box coordinates.
    """
    num_bboxes = len(score_bbox)
    for num in range(num_bboxes):
        bbox_coords = score_bbox[num].split(':')[-1].split(',\n')
        if bbox_coords != ['{}']:
            l_t = float(bbox_coords[0].strip(' array([').strip(']').split(',')[0])
            t_l = float(bbox_coords[0].strip(' array([').strip(']').split(',')[1])
            r_t = float(bbox_coords[1].strip(' [').strip(']').split(',')[0])
            t_r = float(bbox_coords[1].strip(' [').strip(']').split(',')[1])
            r_b = float(bbox_coords[2].strip(' [').strip(']').split(',')[0])
            b_r = float(bbox_coords[2].strip(' [').strip(']').split(',')[1])
            l_b = float(bbox_coords[3].strip(' [').strip(']').split(',')[0])
            b_l = float(bbox_coords[3].strip(' [').strip(']').split(',')[1].strip(']'))
            pts = np.array([[int(l_t), int(t_l)], [int(r_t), int(t_r)], [int(r_b), int(b_r)], [int(l_b), int(b_l)]])

            if np.all(pts) > 0:
                word = crop(pts, image)
                folder = '/'.join(image_name.split('/')[:-1])

                # Ensure output directory exists
                full_output_dir = os.path.join(output_dir, folder)
                if not os.path.isdir(full_output_dir):
                    os.makedirs(full_output_dir)

                try:
                    file_name = os.path.join(full_output_dir, '{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(
                        l_t, t_l, r_t, t_r, r_b, b_r, l_b, b_l))
                    cv2.imwrite(file_name, word)
                    print('Image saved to ' + file_name)
                except:
                    continue

# Path to the CSV file (assuming it's in the same directory as the script)
csv_path = 'data.csv'

# Path to the directory containing test images
test_images_path = 'small_sample/'

# Directory where cropped images will be saved
output_dir = 'E:/College/Hackathon/Amazon ML Challenge/CRAFT/Amazon-ML-Challenge/Crop Words'

# Read CSV
data = pd.read_csv(csv_path)

for image_num in range(data.shape[0]):
    image = cv2.imread(os.path.join(test_images_path, data['image_name'][image_num]))
    image_name = data['image_name'][image_num].strip('.jpg')
    score_bbox = data['word_bboxes'][image_num].split('),')
    generate_words(image_name, score_bbox, image, output_dir)
