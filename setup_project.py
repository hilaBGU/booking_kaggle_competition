import os


def create_project_structure(base_path):
    # Define the folder structure
    folders = [
        "data",
        "notebooks",
        "scripts",
        "results"
    ]

    # Create the folders
    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Created folder: {path}")

    # Create a README.md file
    readme_path = os.path.join(base_path, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write("# Kaggle Project\n\nThis project is for the Booking Challenge competition on Kaggle.")
        print(f"Created file: {readme_path}")

    # Create a main.py file
    main_py_path = os.path.join(base_path, "main.py")
    if not os.path.exists(main_py_path):
        with open(main_py_path, "w") as f:
            f.write(
                """import pandas as pd\n\n# Main pipeline\nif __name__ == "__main__":\n    print("Hello, Kaggle!")""")
        print(f"Created file: {main_py_path}")


if __name__ == "__main__":
    base_dir = os.getcwd()  # Get the current working directory
    create_project_structure(base_dir)
