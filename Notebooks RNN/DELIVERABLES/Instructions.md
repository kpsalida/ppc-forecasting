# README

## CreateFeatureDataFrames

### Files to upload
- Either 48 hours (for the LSTM model) or 24 hours (for the GRU model) in the past with the actual and forecasted prices.
- Also 24 hours ahead file with predictions only

### Dataframes
- **48/24 hours in the past (df_48/df_24)**:
  - `datetime_from`
  - `price`
  - `price1`
  - `price2`
  - `price3`
  
- **24 hours ahead of forecasts (df_last_day)**:
  - `datetime_from`
  - `price1`
  - `price2`
  - `price3`

### Columns Description
- `datetime_from`: the timestamp
- `price`: the actual energy price
- `price1`, `price2`, `price3`: Price forecasts of the 3 forecasters

### Notes
- You will need to upload the initial files.
- Ensure all hours are present.

### Script Features
- Create Temporal Features (hour, dayofweek, month)
- Make Cyclical Feature Encoding for hour, dayofweek, month
- Define if it is a weekend or not (1/0)
- Define if it is a public holiday or not (1/0). It depends on the file `holidays.py` where we define all the public holidays of the year and needs to be updated.

### Important Lines
- **Line 19**: Will be saving the file locally.
- **Line 43**:
  - If you have uploaded a file with more timestamps, then in line 57 of the code you will be able to select the timestamps you want for 48 or 24 hours (df_48) in the past or 24 hours ahead for the 24-hour forecasting (df_last_day).

## GRU_PREDICT / LSTM_PREDICT

### Requirements
- Load the model
- Load the scaler
- Load the dataset with:
  - 48 hours (LSTM) or 24 hours (GRU) history actuals with predictions from 3 forecasters (df_48/df_24)
  - 24 hours of predictions ahead (df_last_day)

### Dataframe Creation
- These files are created through the `CreateFeatureDataframes` Notebook.
- They incorporate weekend and public holiday noting and cyclical feature creation for the hour, dayofweek, and month.

### Predictions
- Predictions for the actual energy price together with the forecasts are given in the dictionary `Predictions`.
- How many hours ahead to predict is passed as a parameter.

### Ensure
- All hourly intervals are present in the files.
- The `df_48/df_24` and the `df_last_day` are continuous in hourly intervals.

## GRU_TRAIN / LSTM_TRAIN

### Files to train GRU and LSTM models respectively
- **GRU MAE** ~ 9.5
- **LSTM MAE** ~ 10.5

## GRU_PREDICT_TEST_<Test_id> / LSTM_PREDICT_TEST_<Test_id>

### Files
- Files with known data for the past and future so as to test the predictions.



