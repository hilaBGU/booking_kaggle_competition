# Booking Kaggle Competition

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
The dataset for this project is hosted on Kaggle and needs to be downloaded separately. To download the dataset:
1. Log in to [Kaggle](https://www.kaggle.com/) with your account credentials.
2. Navigate to the competition page at this link: [Booking Challenge](https://www.kaggle.com/competitions/booking-challenge).
3. Download the dataset files either manually or using the Kaggle API.  
   If you use the API, first ensure that your Kaggle API key is set up correctly. Then, download the dataset by running the appropriate command. Extract the dataset files and place them in a folder named `data` inside the repository.

### 4. Project Structure
Once everything is set up, your repository should look like this:
```
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