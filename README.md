# Booking Kaggle Competition

### Team Name: ### H&R
### Team Members: ### 
Hila Zylfi
Ruben Sasson

This repository contains the code and resources for the Booking Kaggle competition. Due to file size limitations, the dataset is not included in this repository and must be downloaded separately.

## Prerequisites
Before running the project, ensure the following are installed:
- Python 3.7 or higher
- `pip` or `conda` for managing Python packages
- Kaggle API (`pip install kaggle`)

## Setup Instructions
### 1. Clone the Repository
To get started, navigate to the directory on your local machine where you want the project files to be located. Then, clone the repository using the GitHub link:  
[https://github.com/hilaBGU/booking_kaggle_competition.git](https://www.github.com/hilaBGU/booking_kaggle_competition.git).  
Once cloned, navigate into the project directory.

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
The dataset for this project is hosted on Kaggle, and the process of downloading and extracting it has been automated. To set up the dataset:

Run the setup.py script:

```bash
python setup.py
```

This script will:

Download the dataset from Kaggle.
Extract it into the data folder inside the repository.
Install all required Python dependencies.

### 4. Project Structure
Once everything is set up, your repository should look like this:

```bash
booking_kaggle_competition/
├── data/                 # Dataset files (download separately)
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── scripts/              # Utility scripts and functions
├── README.md             # Project instructions
├── main.py               # Main execution script
├── requirements.txt      # Python dependencies
```


### 5. Running the Project
After setting up the environment and downloading the dataset, you can execute the project by running the main script. To do this, locate the `main.py` file in the repository and execute it using Python. Ensure that all dependencies are installed and the dataset files are correctly placed in the `data` folder.

---

## Notes
- Ensure that the dataset files (`train_reviews.csv`, `train_users.csv`, etc.) are located in the `data/` folder before running the project.
- If you encounter any issues, please feel free to open an issue or contact the repository owner.