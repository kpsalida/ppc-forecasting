# Optimizing Next-Day Energy Price Predictions


In collaboration with Big Blue Data Academy and PPC, worked on a short-term data science consulting project for Optimizing Next-Day Energy Price Predictions. The project involved deploying ML models using regression and time series analysis to forecast next-day energy prices. We utilized predictive data from three independent energy price providers  (forecasters), alongside historical price data, to enhance accuracy. This solution will support the PCC Short-Term Market Analysis Unit in improving the precision of their 24-hour energy price predictions.

Duration: 1 month.

## Project Organization
```

├── README.md       <- The top-level README for navigating this project

├── Data            <- The original data from PPC with actual the actual energy prices per hour and the forecasted prices from 3 external providers

├── DataANDMerging  <- Notebooks for creating ONE target file, merging, joining, interpolating for missing timestamps

├── EDA             <- Notebooks for performing Exploratory Data Analysis

├── Notebooks RNN   <- Jupyter notebooks for Training RNN algorithms and providing production files for PREDICTING Next-Hours Energy Prices

├── PPC ENV RNN     <- Environment for running RNN Models

├── Presentation    <- Presentation walking through the project and final conclusions

```

## Data
___

We had to merge various files that held the actual energy prices, together with files providing forecasts for each one of the forecasters.

## DataAndMerging
___

We had to make sure that the timeseries data were consistent with prices per hour. We merged additional files provided by PPC where we had big gaps and also interpolated for small intervals with no data.

## EDA
___

Notebooks where we performed the Exploratory Data Analysis that is depicted in the presentation. More specifically:
-  Monthly Analysis of average actual prices
-  Hourly analysis if average actual energy prices
-  Association with major worldwide events affecting energy prices
-  Comparative analysis of mean deviations of the three forecater energy prices vs. actual prices

## Notebooks RNN
___
Parameters that affecting the outcome of the neural network (NN)
- Incorporated <span style="text-decoration: underline;">temporal features</span> like the time of day, day of week and month. Also flagged days being holidays and weekends
- Hour, DayofWeek and Month were transformed <u>cyclically</u>, so as to ensure that the algorithms understand cyclic recurrence over time
- Experimented with :
       - The architecture of the NN 
       - Lagging of the observations - periodicity dependency
       - Learning dependency of the algorithm from Batch size and epochs 

## Final Prediction
___
!(C:\Users\kathe\Documents\$user\MINE\GitHubProjects\ppc-forecasting\Notebooks RNN\FinalPrediction.png)

## Conclusions
___
|               | Forecaster 1 | Forecaster 2 | Forecaster 3 | Light GBM  | LSTM   | GRU    |
|---------------|--------------|--------------|--------------|------------|--------|--------|
| **MAE**       | 15.53        | 16.33        | 14.61        | **11.87**  | **9.84** | **9.78** |


It is notable that we were able to train models which, by utilizing the predictions of the three forecasters, provide us with better forecasts for future energy prices. We recorded our experiments and results through the online platform <u> Neptune.ai</u>, which greatly assisted us in organizing and leveraging the results.


## Participants
___

- Katerina Psallida
- Kostas Tsagkaropoulos