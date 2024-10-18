from transformers import AutoTokenizer
import evaluate
import numpy as np
import torch
from sklearn.model_selection import KFold

metric = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Function to preprocess text data
def preprocess_text(text, tokenizer):
    # Tokenize the text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    return inputs

# Function to make predictions
def predict_text(model, inputs):
    # Disable gradient calculation
    with torch.no_grad():
        # Make predictions
        outputs = model(**inputs)
        # Get the predicted probabilities
        probabilities = torch.softmax(outputs.logits, dim=1)
        # Get the predicted class (0 or 1)
        predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities


def elaborate_input(data, input_columns, label_column):
    """ return two dataframe (obtained as a subset of the input one): 
    one with the columns that represent the input of the model and
    one with the label column
    Args:
        data: dataframe
        input_columns: list of columns of data to use as input for the model
        label_column: label column
    """

    """
    x_data = []
    for value in data.loc[:, data.columns != label_column].iterrows():
        new_value = []
        for input_column in input_columns:
            new_value.append(value[1][input_column])
            #new_value = new_value + [value[1][input_column]]
        
        x_data.append(new_value)
    x_data = np.array(x_data)
    """
    x_data = data[input_columns].squeeze().tolist() #.loc[:,input_columns].values
    """
    y_data = []
    for value in data[label_column]:
        y_data.append([int(value)])
    y_data = np.array(y_data)
    """
    y_data = data[label_column].squeeze().tolist()
    return x_data, y_data

def elaborate_data_10fold(data, train_index, test_index, iteration,
                          input_columns, label_column):
    """uses index obtained by 10Fold split and the iteration number to identify the validation partition
    return train, validation and test sets according to the selected label
    Args:
        data: dataframe
        train_index: index for the training set, according to 10Fold
        test_index: index fot the test set, according to 10Fold
        iteration: 10Fold's iteration number
        input_columns: list of columns of data to use as input for the model
        label_column: label column
    """

    """use last train fold as validation """
    if iteration == 0:
        i = 8
    else:
        i = iteration - 1

    a, b = list(KFold(n_splits=9).split(train_index))[i]
    val_index = train_index[b]
    train_index = train_index[a]

    x_train_GS, y_train_GS = elaborate_input(data.iloc[train_index, :], input_columns, label_column)
    x_val_GS, y_val_GS = elaborate_input(data.iloc[val_index, :], input_columns, label_column)

    x_test_GS, y_test_GS = elaborate_input(data.iloc[test_index, :], input_columns, label_column)

    return x_train_GS, y_train_GS, x_val_GS, y_val_GS, x_test_GS, y_test_GS
