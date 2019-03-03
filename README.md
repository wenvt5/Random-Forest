# Random-Forest
Forecasting by using random forest in Python sklearn package

## Functions
### Prepare_data
- read data from your local file  
- split to training data and test data in ratio of 0.75:0.25
- Transform the categorical data to quantitative data by using get_dummies
### train_and_predict
- build a forest model with specification of forest_size and tree_level
- fit the model with prepared training data
- predict with the prepared testing data and calculate the prediction accuracy
### visualize_tree
- create a dot file for a random picked tree with export_graphviz
- convert the dot file to a png file
### features_importances
- list the features importance for all the predictors
### plot_importances
- plot the features importance for all the predictors
### __main__
- train the model, pick the top 2 important predictors to rebuild model
