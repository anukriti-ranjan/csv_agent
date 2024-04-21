# CSV Agent

Chat with your CSV.
Get plots, dataframes and more via streamlit app
Can also be used to create a backend API.

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

## OpenAi API key
Create a file .env in root. Add your openai api key. The template should be picked from .env.template

## Run Agent

streamlit run app.py



![csv_agent1](https://github.com/anukriti-ranjan/csv_agent/assets/89630232/b64d320e-3e33-4427-9741-948fabbb8964)
