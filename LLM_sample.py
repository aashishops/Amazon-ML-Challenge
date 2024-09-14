import pandas as pd
import re

# Sample entity_unit_map with more unit variants
entity_unit_map = {
    "width": {"centimetre", "cm", "foot", "millimetre", "mm", "metre", "m", "inch", "yard"},
    "depth": {"centimetre", "cm", "foot", "millimetre", "mm", "metre", "m", "inch", "yard"},
    "height": {"centimetre", "cm", "foot", "millimetre", "mm", "metre", "m", "inch", "yard"},
    "item_weight": {"milligram", "kg", "kilogram", "g", "gram", "ounce", "ton", "lb", "pound"},
    "maximum_weight_recommendation": {"milligram", "kg", "kilogram", "g", "gram", "ounce", "ton", "lb", "pound"},
    "voltage": {"millivolt", "kilovolt", "volt"},
    "wattage": {"kilowatt", "watt"},
    "item_volume": {"cubic foot", "microlitre", "cup", "fluid ounce", "centilitre", "imperial gallon", "pint", 
                    "decilitre", "litre", "ml", "millilitre", "quart", "cubic inch", "gallon"}
}

# Function to detect entity based on unit
def detect_entity(predicted_word):
    if isinstance(predicted_word, str):  # Check if it's a string
        for entity, units in entity_unit_map.items():
            for unit in units:
                if unit in predicted_word.lower():  # Case insensitive match
                    return entity
    return None

# Function to extract numerical value and unit from the text
def extract_value_unit(predicted_word):
    if isinstance(predicted_word, str):  # Ensure predicted_word is a string
        match = re.search(r"(\d+\.?\d*)\s*(\w+)", predicted_word)
        if match:
            value = match.group(1)
            unit = match.group(2)
            return value, unit
    return None, None

# Load CSV data from predicted_words.csv
data = pd.read_csv('predicted_words.csv')  # Load your file

# Group by image and process
grouped_data = data.groupby('image_name')

# Output result
result = {}

for image, group in grouped_data:
    entities = {}
    print(f"Processing Image: {image}")  # Debug print to show processing images
    for _, row in group.iterrows():
        predicted_word = row['predicted_word']
        confidence = row['confidence_score']
        print(f"  Predicted Word: {predicted_word}, Confidence: {confidence}")  # Debug print for each word
        
        entity = detect_entity(predicted_word)
        if entity:
            value, unit = extract_value_unit(predicted_word)
            if value and unit:
                entities[entity] = f"{value} {unit}"
                print(f"    Detected Entity: {entity}, Value: {value}, Unit: {unit}")  # Debug print for detected entities
    
    result[image] = entities

# Print result
for image, entity_data in result.items():
    if entity_data:
        print(f"Image: {image}")
        for entity, value in entity_data.items():
            print(f"  {entity}: {value}")
    else:
        print(f"Image: {image} - No valid entities detected")
