# Example (HELOC Survival)

* Piecewise Model 
  * the purpose of the piecewise model is because the data is limited for some data with age > N, so it is not stable, so we assume it has constant hazard rate for 2nd piece.

* Assumption:
  * logrank test: to check if the groups are seperated properly (for example, different group have different survival curves)

* Constant hazard rate:
  * using KaplanMeierFitter
  * since the data after age X the boservations are limited, so we use constant hazard rate to estimate survival rate
  * to get X value, check the 1-year moving window hazard rate standard deviation, tweak the proper cut-off month, then calculate average month over month hazard rate after cut-off months as constant hazard rate

* Weibull-AFT model
  * for 1st piece, fit weilbull aft model
  * suitability check: log t and S0 reverse should be linear relationship
  * modify the dataset to include all accounts with age >= cut-off months as right censord 
  * split data to training/testing/validation dataset
  * fit the model in training dataset with different variable / variable combinations
  * check the variable p-value (wald-test/likelihood ratio) to see if variables are significant
  * do variable selection based on brier score on validation dataset

* Some Python functions:
  * logrank test: from lifelines.statistics import logrank_test  
  * suitability check: 
    ```
    S0_reverse = km_data.KM_Survival.apply(lambda x: np.log(-1*np.log(x)))
    log_T = km_data.Month.apply(lambda x: np.log(int(x)))
    px.line(data, x='log_T', y= 'S0_reverse'
    ```
