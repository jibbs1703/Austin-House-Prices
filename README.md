# Austin-House-Prices
House pricing is based on a variety of factors and is a vital aspect of the real estate industry. It is of importance to various stakeholders in the industry, such as real estate professionals, investors, and homebuyers. By gaining insights and understanding the factors that influence property values, stakeholders can make well informed decisions and achieve more successful outcomes in the ever competitive housing market.

## Dataset
The Dataset for this study was obtained from [Kaggle](https://www.kaggle.com/datasets/ericpierce/austinhousingprices/data?select=austinHousingData.csv). The dataset collected information on house prices in Austin, Texas over four years from 2018 to 2021. The dataset includes 15,171 listings with 45 features. Since the Austin Housing dataset spans across four years - 2018, 2019, 2020 and 2021, the price prediction must deal with the variation in prices in the dataset due to multiple sale years and this is done by making the prices uniform as 2023 prices, accounting for inflation. The effect of inflation can be captured through a variety of metrics, however, for this analysis, the consumer price index (CPI) is the chosen metric as it adjusts the value of prices in previous time periods to the current time period. The conversion factor for price adjustment is obtained from the [Minneapolis Federal Reserve Bank](https://www.minneapolisfed.org).

## Chronology 
- Import necessary libraries and datasets.
- Identify datatypes present and make appropriate conversions.
- Preprocess dataset - deal with missing values scale numeric features and label categorical features accordingly.
- Split data into the features and target.
- Split data into train and test datasets.
- Train machine learning models on train dataset and check training metric.
- Use trained model on test dataset to make house price prediction and checking metric.
- Save price predictions model.

## Model Results


## Author(s)
- **Abraham Ajibade** [Linkedin](https://www.linkedin.com/in/abraham-ajibade-759772117)
- **Boluwtife Olayinka** [Linkedin](https://www.linkedin.com/in/ajibade-bolu/) 