import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report
from Utils2 import preprocessing, model_performances

seed = 222
np.random.seed(seed) # numpy seed
tf.random.set_seed(seed) # works for all devices (CPU and GPU)

dataset_name = 'ConvAbuse'
id_terms = [['asshole', 'pussy', 'fuck', 'stupid', 'sex', 'idiot', 'kill', 'hate', 'bitch', 'slut'], ['mention', 'dead', 'english', 'nope', 'hungry', 'ever', 'old', 'first', 'meet', 'keep']]

#dataset_name = 'HS-Brexit'
#id_terms = [['paki', 'deport', '#trump2016', 'islam', 'obama', 'muslim', 'world', 'illegal', 'invasion', 'white'], ['blame', 'cause', 'terrorism', 'might', 'another', 'attack', 'watch', '#remain', 'bad', '#brexitvote']]

#dataset_name = 'MD-Agreement'
#id_terms = [['shitty', 'cunt', 'fuck', 'stfu', 'hitler', 'bullshit', 'fuckin', 'niggas', 'asshole', 'pedophile'], ['insurance', 'ballot', 'anger', 'difficult', 'wall', 'cancel', 'across', 'replace', 'company', '#anonymous']]

#dataset_name = 'ArMIS'
#id_terms = [['الفسويات', 'وقحات', 'قذرات', 'النار', 'متسلطات', 'النسويات', 'ايش', 'ههه', 'رخيصات', 'وهي'], ['نسويه', 'تقول', 'يقول', 'كانوا', 'لأن', 'العنف', 'النسويه', 'علي', 'المراه', 'يأخذ']]

# ________________________________________Utils ___________________________________________________
if not os.path.exists('./'+dataset_name+'/Unimodal/predictions'):
    os.makedirs('./'+dataset_name+'/Unimodal/predictions')

if not os.path.exists('./'+dataset_name+'/Unimodal/performances'):
    os.makedirs('./'+dataset_name+'/Unimodal/performances')

path_models = './'+dataset_name+'/Unimodal/models'
file_out = './'+dataset_name+'/Unimodal/performances/Text_results_10Fold.txt'
predictions_csv_path = './Unimodal/'+dataset_name+'/predictions/Text_pred_10Fold.csv'

file = open(file_out, 'a+')
file.truncate(0)  # erase file content
file.close()


label_column = "hard_label"
input_columns = ['text_USE']
threshold = 0.5
embed_size = 512  # 512-length array with Universal Sentence Encoder algorithm
batch_size = 64
epochs = 100

# ________________________________________Load Data ___________________________________________________
print("Loading data...")
from Utils import preprocessing as pr
folder = "./Data/"
# ________________________________________load training data ___________________________________________________

train_df = pd.read_json(folder + dataset_name + "_dataset/" + dataset_name + "_train.json", orient='index')
train_df = pr.get_dataset_labels(train_df)

train_df['text_USE'] = preprocessing.use_preprocessing(train_df, 'original_text')
train_df['file_name'] =train_df.index

# ________________________________________Utils ___________________________________________________
if not os.path.exists('./'+dataset_name+'/Unimodal/predictionsTest'):
    os.makedirs('./'+dataset_name+'/Unimodal/predictionsTest')

if not os.path.exists('./'+dataset_name+'/Unimodal/performancesTest'):
    os.makedirs('./'+dataset_name+'/Unimodal/performancesTest')


path_predictions = './'+dataset_name+'/Unimodal/predictions/'
path_performances = './'+dataset_name+'/Unimodal/performances/'
file_out = path_performances + 'text_results_10Fold.txt'
predictions_csv_path = path_predictions + 'text_pred_10Fold.csv'

for path in [path_models, path_predictions, path_performances]:
    if not os.path.exists(path):
        os.makedirs(path)

file = open(file_out, 'a+')
file.truncate(0)  # erase file content
file.close()

print("training model on training data 10Fold ...")
# ________________________________________train model on training data 10Fold________________________________________
kf = KFold(n_splits=10, shuffle=False)

iteration = 0
real_values = np.array([])
predict_values = np.array([])
ids = np.array([])

for train_index, test_index in kf.split(train_df):  # split into train and test
    preprocessing.set_seed(iteration)
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.elaborate_data_10fold(train_df,
                                                                                        train_index,
                                                                                        test_index,
                                                                                        iteration,
                                                                                        input_columns,
                                                                                        label_column)
    model, history = model_performances.get_trained_model(x_train, 
                            y_train, 
                            x_val, 
                            y_val,
                            input_shape=embed_size, 
                            activation_function='LeakyReLU', 
                            neurons=embed_size/2, 
                            dropout=0.2, 
                            epochs=100)
    iteration = iteration + 1

    # make prediction on GS
    pred = model.predict(x_test, batch_size=batch_size)

    predict_values = np.append(predict_values, pred)
    real_values = np.append(real_values, y_test)
    ids = np.append(ids, train_df.iloc[test_index, :]['file_name'].tolist())

    result_df = train_df.iloc[test_index, [0, 1]]
    result_df['score_col'] = pred

    # write on file
    file = open(file_out, "a+")
    file.write('\n\nITERAZIONE ' + str(iteration) + '\n')
    file.write(json.dumps(model_performances.compute_confusion_rates(result_df, 'score_col', 'hard_label', threshold)))
    file.write('\n') 
    file.write(classification_report(result_df['hard_label'].values, (result_df['score_col']>threshold).astype(int).values, target_names=['not_hate','hate']))
    file.close()

# results dataframe, save predictions
result_df = pd.DataFrame({'id': ids, 'real': real_values.astype(int), 'pred': predict_values})
result_df.to_csv(predictions_csv_path, index=False, sep='\t')

# Overall metrics write on file
file = open(file_out, "a+")
file.write('\n\n10 Fold Results ' + str(iteration) + '\n')
file.write(json.dumps(model_performances.compute_confusion_rates(result_df, 'pred', 'real', threshold)))
file.write('\n') 
file.write(classification_report(result_df['real'].values, (result_df['pred']>threshold).astype(int).values, target_names=['not_hate','hate']))
file.write('\n AUC:') 
file.write(str(model_performances.compute_auc(result_df['real'].values, result_df['pred'].values)))
file.close()

# ________________________________________load Test + SYN data ___________________________________________________
# Load Test and preprocessing
test_df =  pd.read_json(folder +"/"+dataset_name+"_dataset/"+dataset_name+"_test.json", orient='index')
test_df = pr.get_dataset_labels(test_df)
test_df['text_USE'] = preprocessing.use_preprocessing(test_df, 'original_text')

test_df['file_name']=test_df.index
x_test, y_test = preprocessing.elaborate_input(test_df, input_columns, label_column)

syn_df = pd.read_csv('Syn_df/'+dataset_name+ '_syn.csv', sep='\t')
# ilsolo la seconda metà di SYN
syn_df = syn_df[int(len(syn_df)/2):]
syn_df['text_USE'] = preprocessing.use_preprocessing(syn_df, 'original_text')
syn_df['file_name'] = syn_df.index 

x_syn, y_syn = preprocessing.elaborate_input(syn_df, input_columns, label_column)

res_syn = syn_df[['file_name', 'hard_label']].copy()
res_test = test_df[['file_name', 'hard_label']].copy()

#_________________________Models on Traing-Test Data _____________________________________

MODELNAMES = ['unimodal_text_v{}'.format(i) for i in range(10)]
path_models = './'+dataset_name+'/Unimodal/models/Bias/'
model_name = 'unimodal_text_captions'

if not os.path.exists(path_models):
    os.makedirs(path_models)

print("train model on training data 10Fold for Test-SYN")
# ________________________________________train model on training data 10Fold________________________________________
kf = KFold(n_splits=10, shuffle=False)
iteration = 0

for train_index, val_index in kf.split(train_df):
    preprocessing.set_seed(iteration)
    MODELNAME = MODELNAMES[iteration]

    x_train, y_train = preprocessing.elaborate_input(train_df.iloc[train_index, :], input_columns, label_column)
    x_val, y_val = preprocessing.elaborate_input(train_df.iloc[val_index, :], input_columns, label_column)

    model, history = model_performances.get_trained_model(x_train, 
                            y_train, 
                            x_val, 
                            y_val,
                            input_shape=embed_size, 
                            activation_function='LeakyReLU', 
                            neurons=embed_size/2, 
                            dropout=0.2, 
                            epochs=epochs)

    # save each model once for all
    model.save(path_models + MODELNAME)
    iteration = iteration + 1

# ________________________________________Load models and test________________________________________
x_syn, y_syn = preprocessing.elaborate_input(syn_df, input_columns, label_column)

MODELNAMES = ['unimodal_text_v{}'.format(i) for i in range(10)]
path_models = './'+dataset_name+'/Unimodal/models/Bias/'
path_results_test = './'+dataset_name+'/Unimodal/predictions/predictions_test/'
path_results_syn = './'+dataset_name+'/Unimodal/predictions/predictions_syn/'
path_performances = './'+dataset_name+'/Unimodal/performances'
model_name = 'unimodal_text'

file_out_test = "./"+dataset_name+"/Unimodal/performances/text_Results_Test.txt"
file_out_syn = "./"+dataset_name+"/Unimodal/performances/text_Results_Syn.txt"
file_out_bias = "./"+dataset_name+"/Unimodal/performances/text_Results_Bias.txt"

for path in [path_results_test, path_results_syn, path_performances]:
    if not os.path.exists(path):
        os.makedirs(path)

for file_name in [file_out_test, file_out_syn, file_out_bias]:
    file = open(file_name, 'a+')
    file.truncate(0)  # erase file content
    file.close()

import keras
# ______________________________ retrieve saved 10 models and make predictions _________________________________
print("loading moden and testing...")
kf = KFold(n_splits=10, shuffle=False)
#syn_folds = kf.split(syn_df)
pred_syn=[]
syn_10_df=pd.DataFrame()
for MODELNAME in MODELNAMES:
    # LOAD MODEL
    loaded_model = keras.models.load_model(path_models + MODELNAME)

    # make prediction on test
    predict_values = loaded_model.predict(x_test, batch_size=batch_size)
    res_test[MODELNAME] = predict_values

    # make prediction on syn
    predict_values_s = loaded_model.predict(x_syn, batch_size=batch_size)
    res_syn[MODELNAME] = predict_values_s

    # performances on splitted Syn
    syn_10_df['label_'+MODELNAME]=list(res_syn[label_column].values)
    syn_10_df[MODELNAME]= list(res_syn[MODELNAME].values)
    syn_10_df['file_name_'+MODELNAME]=list(res_syn['file_name'].values)


res_test.to_csv(path_results_test + "baseline_" + model_name + "_scores.tsv", sep="\t", index=False)
res_syn.to_csv(path_results_syn + "baseline_" + model_name + "_SYN_scores.tsv", sep="\t", index=False)


model_performances.confusion_rates_on_file(file_out_test, res_test, MODELNAMES, label_column, threshold)
model_performances.confusion_rates_on_file(file_out_syn, res_syn, MODELNAMES, label_column, threshold)
model_performances.confusion_rates_on_file_10Fold_syn(file_out_syn, syn_10_df, MODELNAMES, threshold)

from ast import literal_eval
syn_df['tokens'] = syn_df['tokens'].apply(lambda x: literal_eval(x))


def contains_word(tokens, word):
    return any(word in tokens for word in [word])

for term in [item for sublist in id_terms for item in sublist]:
    syn_df[term]= ''
    syn_df[term]= syn_df['tokens'].apply(lambda x: contains_word(x, term))

def clear_identity_list(identity_list, df):
    """ Take a list of identity elements (tags or temrs), and a dataframe in which every element in the list have a
    corresponding column indicating its presence in the meme.
    Returns a list, subset of identity_list, with the only element to which at least a misogynous and at least a non
     misogynous meme are associated"""
    #At least one misogynous and one not misogynous per tag:
    to_remove=[]
    for tag in identity_list:
        if len(df.loc[df[tag]==True,'hard_label'].value_counts())<2:
            to_remove.append(tag)
    for tag in to_remove:
        identity_list.remove(tag)
    return identity_list

from Utils2 import load_data
Identity_Terms = clear_identity_list([item for sublist in id_terms for item in sublist], syn_df)

res_syn = res_syn.merge(syn_df.drop(columns=['original_text', 'hard_label', 'soft_label_0', 'soft_label_1',
       'disagreement', 'sentences', 'tokens_lists', 'tokens_list',
       'lemmi_text', 'tokens', 'term_present', 'text_USE']),
                        how='inner', on='file_name')

res_syn['misogynous'] = res_syn['hard_label']
res_test['misogynous'] = res_test['hard_label']

# _________________________________Compute Bias Metrics_____________________________________________________
# Computes per-subgroup metrics for all subgroups and a list of models.
model_performances.compute_bias_metrics_for_models(res_syn,
                                                   Identity_Terms,
                                                   MODELNAMES,
                                                   label_column)

model_performances.bias_metrics_on_file(file_out_bias, res_test, res_syn, Identity_Terms, MODELNAMES, label_column)

# _________________________________Compute Bias Metrics_____________________________________________________
# Computes per-subgroup metrics for all subgroups and a list of models.

model_performances.bias_metrics_on_file(file_out_bias, res_test, res_syn, [item for sublist in id_terms for item in sublist], MODELNAMES, label_column)