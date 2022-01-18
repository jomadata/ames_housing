# Ames Housing Data and Kaggle Challenge
### by Joomart Achekeev

## Introduction

The initial task of the project was to create a linear regression model that will predict house prices depending on a number of parameters, on  the Ames Housing Dataset. This initial datset is comprised of over 2000 buy-sell transactions on property in small town of Ames,Iowa. The data was collected between 2006 and 2010 and is widely used to train machine learning models for educational purposes.

## 01 EDA and Cleaning 

At first the data basic dataset exploration yeilded information that this data has 2000 rows and 81 columns. 
out of 81 columns 43 columns are non-numeric, 26 columns have null at least one cell filled with null value. 
The first step was to camel case the column names for future convenience. 
Over the next couple of steps null values are filled with local average or substitution of null values in category columns with 'no_alle', 'no_basement', etc.
Data outliers have been slightly cleaned in the following step.
Next stage was to conduct an indepth EDA, by looking at certain correlations, identifying areas with more expensive properties, price histogram, and identification of the most important parameters for building of a model.

## 02. Preprocessing and Features Engineering

In this section the data was carefully prepared for modelling via: 
1) numerization of the quality columns.
2) interaction features' columns have added for overall, exterior, basement and garage quality and condition
3) dummifying of categorical variables

## 03. Modelling

For this section three models have been built to see which model yeilds best reuslts:
1. Basic model with numeric data extracted from the initial cleaned dataframe (only numeric data)
0.872755748555276, 0.8888865723894831, 0.8616352367013504
This model resulted in 86.16% explainability, meaning that the model can explain only 86.16% of the results. As this is the basic model that did not have all the data provided from the dataset the next model was built. 

2. A middle complexity model that was fed not only the basic numeric data, but also quality columns data that where numericaly categorized
0.890656138967306, 0.8995652468169439, 0.8772694371856385
This model had an improvement over the basic model with explainability of 87.72%, indicating that we where moving in the right direction in terms of complexity/explainability ratio.

3. The last model incoporated all the possible data that was received during the cleaning and preprocessing stages. 0.9361785706760581, 0.918175065754953, 0.8943265414478188
As it can be seen the model yeilded the highest result of 89.43% explainability, yeilding the best result out of all three models.

## Conclusions

As seen from the modeling the more complex model yields the best results in terms of the base data explaining the highest percentage of target variables. It can provide ~89% of explainability. But nevertheless it is needed to understand that this is a model, it can show general trends in the data, but can lack seeing fine details that sometimes decided the whole picture. Thus the models should only be comprehended in colaboration with understading of their limitations.
