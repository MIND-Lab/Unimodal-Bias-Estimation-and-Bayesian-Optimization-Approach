# IMPORT
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import gc
import random

identity_tag_path = './Data/Categories.txt'


def get_dataset_labels(df, columns = ['original_text','hard_label','soft_label_0','soft_label_1', 'disagreement']):
  """
  df: dataframe to elaborate
  colums: list of output columns
  ______________________________
  Extract two columns from the soft-label column to represent disagreement on the positive and negative label.
  Add a "disagreemen" column with boolean values (1 for agreement, 0 for disagreement)
  Rename the column "text" in "original text" to distiguish with the token-column "text"
  """
  df['soft_label_1']= df['soft_label'].apply(lambda x: x['1'])
  df['soft_label_0']= df['soft_label'].apply(lambda x: x['0'])
  df['disagreement'] = df['soft_label_0'].apply(lambda x : int(x==0 or x==1))
  df.rename({'text': 'original_text'}, axis=1, inplace=True)
  return df[columns]

def use_preprocessing(df, column):
    """Compute the embedding via the Universal Sentence Encoder algorithm
    for every sentence in the given column using TensorFlow 2.x
    Args:
        df: Dataframe
        column: column name to identify data to process
    """
    # Disabilita l'esecuzione eager se necessario
    # tf.compat.v1.disable_eager_execution()

    # Configura l'utilizzo della GPU se necessario
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)

    # Carica il modulo Universal Sentence Encoder da TensorFlow Hub
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    embed = hub.KerasLayer(module_url)

    # Suddivide il DataFrame in parti per evitare problemi di memoria
    dfs = np.array_split(df, 10)
    text_embeddings = []

    # Processa ogni segmento del DataFrame
    for x in dfs:
        text_embedding = embed(x[column])
        text_embeddings.extend(text_embedding.numpy().tolist())

    # Libera memoria se necessario
    gc.collect()

    return text_embeddings


def elaborate_input(data, input_columns, label_column):
    """ return two dataframe (obtained as a subset of the input one): 
    one with the columns that represent the input of the model and
    one with the label column
    Args:
        data: dataframe
        input_columns: list of columns of data to use as input for the model
        label_column: label column
    """
    x_data = []
    for value in data.loc[:, data.columns != label_column].iterrows():
        new_value = []
        for input_column in input_columns:
            new_value = new_value + value[1][input_column]
        
        x_data.append(new_value)
    x_data = np.array(x_data)

    y_data = []
    for value in data[label_column]:
        y_data.append([int(value)])
    y_data = np.array(y_data)

    return x_data, y_data

def elaborate_input_bert(data, input_columns, label_column):
    """ return two dataframe (obtained as a subset of the input one): 
    one with the columns that represent the input of the model and
    one with the label column
    Args:
        data: dataframe
        input_columns: list of columns of data to use as input for the model
        label_column: label column
    """
    x_data = data[input_columns].squeeze().tolist() #.loc[:,input_columns].values
    y_data = data[label_column].squeeze().tolist()
    return x_data, y_data

def elaborate_data_10fold_bert(data, train_index, test_index, iteration,
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

    x_train_GS, y_train_GS = elaborate_input_bert(data.iloc[train_index, :], input_columns, label_column)
    x_val_GS, y_val_GS = elaborate_input_bert(data.iloc[val_index, :], input_columns, label_column)

    x_test_GS, y_test_GS = elaborate_input_bert(data.iloc[test_index, :], input_columns, label_column)

    return x_train_GS, y_train_GS, x_val_GS, y_val_GS, x_test_GS, y_test_GS

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


def get_tags_list():
    """ return the list of tags """
    with open(identity_tag_path, 'r') as fd:
        for line in fd:
            tag_list = line.lower().split('], ')[0].replace("'", "").strip('][').split(', ')
    return tag_list


def tag_embedding():
    """ Read tags list and compute their embeddings with USE 
    Return a dataframe with two columns:
        - tag: the tag name
        - tags_USE: embedding representation of the tag"""
    # read tags
    """
    with open(identity_tag_path, 'r') as fd:
        for line in fd:
            tmp = line.lower().split('], ')[0].replace("'", "").strip('][').split(', ')
    """
    #compute USE for each tag
    category_embedding = pd.DataFrame()
    category_embedding['tag'] = get_tags_list()
    category_embedding['tags_USE'] = use_preprocessing(category_embedding, 'tag')
    return category_embedding


def meme_tag_embedding(data, tag_column):
    """ Compute for each meme its embedding obtained through the mean of tags'embedding """
    category_embedding = tag_embedding()
    mean = []
    for index, row in data.iterrows():
        tags_emb = []
        for tag in data.loc[index,tag_column].split(' '):
            tags_emb.append(category_embedding.loc[category_embedding['tag'] == tag, 'tags_USE'].values[0])
        mean.append(np.mean(tags_emb, axis = 0).tolist())
    return mean

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed) # numpy seed
    tf.random.set_seed(seed) # works for all devices (CPU and GPU)
