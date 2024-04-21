# csv_agent

## For Unix-based systems:
### Create a virtual environment named 'venv' (you can name it anything)
python3.10 -m venv venv

### Activate the virtual environment
source venv/bin/activate

## For Windows:
### Create a virtual environment named 'venv' (you can name it anything)
python3.10 -m venv venv

### Activate the virtual environment
venv\Scripts\activate.bat

## Ensure pip, setuptools, and wheel are up to date
pip install --upgrade pip setuptools wheel

## Install the packages from requirements.txt
pip install -r requirements.txt

## CSV Data
Put your CSV files in the data folder. Make sure to name the folder 'data' and place it inside the root folder

## Run Agent

streamlit run app.py

