import json


# Save train_categories to a JSON file
def save_categories_to_file(categories, filename):
    with open(filename, 'w') as f:
        json.dump(categories.tolist(), f)


# Load train_categories from a JSON file
def load_categories_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)
