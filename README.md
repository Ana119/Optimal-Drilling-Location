# Drilling Project


## Description
This repository containt the code for the paper title "A New Method for Determination of Optimal Borehole Drilling Location Considering Drilling Cost Minimization and
Sustainable Groundwater Management".

## Experimental Environment
python libraries, 
including 
- NumPy, 
- SkLearn,
- Scipy,
- pandas
- matplotlib
- pickle
- seaborn
- xgboost

## Steps to build

- train_depth_model is a pre-trained model to predict the drilling depth using chained LSTM 
- train_soil_model. is a pretrained ensemble model to classify the soil color and land layer. The model is an ensemble of SVM, GNB, GBM, and RF
- train_water_model is a pre-trained model to predict ground water level using chained LSTM  
- Optimization.py is a pythond scrip to find optimal location utilizing the above pretrained models in a given geographical area.


## Archtecture of the drilling project
![image](https://user-images.githubusercontent.com/106262211/214262805-4aca910c-ee57-49df-87d3-97a842aaeef7.png)
