import os
import json

# Load kaggle.json (in same folder)
with open("kaggle.json", "r") as f:
    creds = json.load(f)

os.environ['KAGGLE_USERNAME'] = creds['username']
os.environ['KAGGLE_KEY'] = creds['key']

# Download
os.system("kaggle datasets download -d nuancelabs/nslkdd -p data/ --unzip")
