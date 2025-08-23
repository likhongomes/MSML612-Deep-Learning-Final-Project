## Running the project
1. In Google Drive create a folder for the project
2. Add tabnet_model.py located in the repository's Notebook folder to the created folder in Google Drive
3. In the Google Drive folder, create a folder called Data and add the cleaned_parking_violations_v3_daily_aggregate.csv and daily)timeseries_parking_violations_v2.csv located in the CleanData folder in the repository to that Google Drive Data folder you just created
4. In the Google Drive folder you just created, create a notebook
5. In the notebook at cell 1 do:
    - from google.colab import drive
    - drive.mount('/content/drive')
6. In cell 2 do:
    - import sys
    - sys.path.append(PATH_TO_CURRRENT_DIR)
7. In cell 3 do:
    - import tabnet_model
8. Running cell 3 will run the baselines and TabTransformer model
