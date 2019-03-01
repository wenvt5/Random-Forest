import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt


def prepare_data(source_file, target_column_name, test_size):
    data = pd.read_csv(source_file)
    labels = data[target_column_name]
    # Remove the labels from the features;axis 1 refers to the columns
    features= data.drop(target_column_name, axis = 1)
    # convert categorical data to numerical data in '0' and '1' form
    features = pd.get_dummies(features)
    feature_list = features.columns
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
    return [train_features, test_features, train_labels, test_labels, feature_list]


def train_and_predict(train_features, test_features, train_labels, test_labels, forest_size, tree_level):
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = forest_size, random_state = 42, max_depth = tree_level)
    # Train the model on training data
    rf.fit(train_features, train_labels);
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    return rf


def visualize_tree(rf,random_pick,dot_file_name,png_file_name):
    # Pull out one tree, such as number 5 or number 8, from the forest
    tree = rf.estimators_[random_pick]
    # Export the image to a dot file
    export_graphviz(tree, out_file = dot_file_name, feature_names = features.columns, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file(dot_file_name)
    # Write graph to a png file
    graph.write_png(png_file_name)


def features_importances(rf,feature_list):
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    return feature_importances


def plot_importance(rf,feature_list):
    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    feature_importances_list = features_importances(rf,feature_list)
    x_values = list(range(len(feature_importances_list)))
    # Make a bar chart
    plt.bar(x_values, list(rf.feature_importances_), orientation = 'vertical')
    # Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')


def main():
    [train_features, test_features, train_labels, test_labels,feature_list] = prepare_data("temps.csv","actual",0.25)
    rf = train_and_predict(train_features, test_features, train_labels, test_labels, 1000, 3)
    plot_importance(rf,feature_list)
    feature_importances_list = features_importances(rf,feature_list)
    # New random forest with only the top 2 most important variables
    top_2_importances = [feature_importances_list [0][0],feature_importances_list [1][0]]
    train_important = train_features[top_2_importances]
    test_important = test_features[top_2_importances]
    rf_most_important = train_and_predict(train_important, test_important, train_labels, test_labels, 1000, 3)


if __name__ == '__main__':
    main()
