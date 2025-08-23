## Running the project
1. In Google Drive create a folder for the project called "MSML612 Project" at "/content/drive/MyDrive/MSML612 Project"
2. Add tabnet_model.py located in the repository's Notebook folder to the created folder in Google Drive
3. In the Google Drive folder, create a folder called Data and add the daily_timeseries_parking_violations_v2.csv located in the CleanData folder in the repository to that Google Drive Data folder you just created
4. In the Google Drive folder you just created, create a notebook
5. In the notebook at cell 1 do:
    - from google.colab import drive
    - drive.mount('/content/drive')
6. In cell 2 do:
    - import sys
    - sys.path.append(PATH_TO_CURRENT_DIR)
7. In cell 3 do:
    - import tabnet_model
8. Running cell 3 will run the baselines and TabTransformer model

![PHOTO-2025-08-22-19-13-11](https://github.com/user-attachments/assets/0178b900-a0f0-4abf-8734-d7a8985fc613)

## To run evaluation notebook:
1. Open a new Google Colab notebook
2. Update tabnet_evaluation.py lines 22 & 24 to reflect the names of your project folder and outputs folder
3. Upload tabnet_evaluation.py to Files in the Google Colab notebook
4. In the notebook at cell 1 do:
    - from google.colab import drive
    - drive.mount('/content/drive')
5. In cell 2 do:
    - import sys
    - sys.path.append(PATH_TO_CURRENT_DIR)
6. In cell 3 do:
    - import tabnet_evaluation
7. Running cell 3 will save the evaluation plots to a folder called "Eval_Notebook_Artifacts" within the output folder

Note: These outputs have also been uploaded to this repo in the Eval_Notebook_Outputs folder
