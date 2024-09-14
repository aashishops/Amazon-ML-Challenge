import pandas as pd

# Load the CSV file
csv_file = "predicted_words.csv"  # Path to your CSV file
df = pd.read_csv(csv_file)

# Create a dictionary to store image_name and combined predicted words
image_text_dict = {}

# Iterate through each row in the dataframe
for index, row in df.iterrows():
    image_name = row['image_name']
    predicted_word = str(row['predicted_word'])
    
    # Extract the base image name (remove _1, _2, etc.)
    base_image_name = image_name.rsplit('_', 1)[0]
    
    # Append the predicted word to the corresponding base image
    if base_image_name in image_text_dict:
        image_text_dict[base_image_name] += ' ' + predicted_word
    else:
        image_text_dict[base_image_name] = predicted_word

# Print the results
for image_name, combined_text in image_text_dict.items():
    print(f"Image: {image_name}.jpg")
    print(f"Combined Text: {combined_text}\n")
