# Model Card

Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random forest classifier is used as a classifier. Random forests are an ensemble learning method for classification, 
regression and other tasks that operates by constructing a multitude of decision trees at training time. For 
classification tasks, the output of the random forest is the class selected by most trees.
Default configuration were used for training.

## Intended Use

This model should be used to predict the category of the salary of a person based on personal features.

## Training Data

Source of data https://archive.ics.uci.edu/ml/datasets/census+income . 
80% of the data is used for training using stratified KFold.

## Evaluation Data

Source of data https://archive.ics.uci.edu/ml/datasets/census+income .
20% of the data is used to validate the model on unseen data.

## Metrics

The model was evaluated using Accuracy score, F1 score, Precision and Recall. Accuracy was 0.823.

## Ethical Considerations
The model performance was calculated on data slices. This drives to a model that may potentially bias people on profession or gender. 
Therefore, this study is just for education purposes and further investigation might be needed. 

## Caveats and Recommendations

Data imbalance in some features might need further investigation as it causes feature bias. 