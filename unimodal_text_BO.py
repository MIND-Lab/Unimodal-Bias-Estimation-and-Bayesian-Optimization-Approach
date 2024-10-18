import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#disable warnings
import logging
import os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from Utils import preprocessing, model_performances, adjusted_performances

import time
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from collections import namedtuple

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
if not os.path.exists('./'+dataset_name+'/Mitigated/predictions'):
    os.makedirs('./'+dataset_name+'/Mitigated/predictions')

if not os.path.exists('./'+dataset_name+'/Mitigated/performances'):
    os.makedirs('./'+dataset_name+'/Mitigated/performances')

path_models = './'+dataset_name+'/Mitigated/models'
file_out = './'+dataset_name+'/Mitigated/performances/Text_results_10Fold.txt'
predictions_csv_path = './Mitigated/'+dataset_name+'/predictions/Text_pred_10Fold.csv'

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
folder = "./Data/"
# ________________________________________load training data ___________________________________________________

train_df = pd.read_json(folder + dataset_name + "_dataset/" + dataset_name + "_train.json", orient='index')
train_df = preprocessing.get_dataset_labels(train_df)

train_df['text_USE'] = preprocessing.use_preprocessing(train_df, 'original_text')
train_df['file_name'] =train_df.index

# ________________________________________Utils ___________________________________________________
path_predictions = './'+dataset_name+'/Mitigated/predictions/'
path_performances = './'+dataset_name+'/Mitigated/performances/'
file_out = path_performances + 'text_results_10Fold_BO.txt'
predictions_csv_path = path_predictions + 'text_pred_10Fold_BO.csv'

for path in [path_models, path_predictions, path_performances]:
    if not os.path.exists(path):
        os.makedirs(path)

file = open(file_out, 'a+')
file.truncate(0)  # erase file content
file.close()


# ________________________________________load Test + SYN data ___________________________________________________
# Load Test and preprocessing
test_df =  pd.read_json(folder +"/"+dataset_name+"_dataset/"+dataset_name+"_test.json", orient='index')
test_df = preprocessing.get_dataset_labels(test_df)
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
MODELNAMES = ['unimodal_text_BO_v{}'.format(i) for i in range(10)]
path_models = './'+dataset_name+"/Mitigated/models/"
path_predictions= './'+dataset_name+"/Mitigated/predictions/"
path_performances = './'+dataset_name+"/Mitigated/performances"
model_name = 'unimodal_text_BO'

results_txt_path = path_performances + 'BO_results.txt'

for path in [path_models, path_predictions, path_performances]:
    if not os.path.exists(path):
        os.makedirs(path)

file = open(results_txt_path, 'a+')
file.truncate(0)  # erase file content
file.close()

# ________________________________________Defining BO________________________________________

# define a search space as a List
search_space = list()
search_space.append(Real(0.00001, 0.1, 'log-uniform', name='lr'))
search_space.append(Real(1e-8, 1, 'log-uniform', name='epsilon'))
search_space.append(Integer(64, embed_size, name='neurons'))
search_space.append(Categorical(['sigmoid', 'relu', 'tanh', 'LeakyReLU'], name='activation_function'))
search_space.append(Real(0.1, 0.6, 'log-uniform', name='dropout'))

default_parameters={"lr":0.00001,
        "epsilon": 1e-7,
        "neurons": 512, 
        "activation_function": 'LeakyReLU',
        "dropout": 0.2
    }

# distinguish variable parameter from fixed parameters in 10Fold execution
VariableParams = namedtuple('VariableParams', 'lr epsilon neurons activation_function dropout')
FixedParams = namedtuple('FixedParams', 'iteration dataset syn_data train_index test_index')

def fitness(v_args, f_args):
    """  A function that will be called by the search procedure. 
    This is a function expected by the optimization procedure later and takes a model and set of specific
    hyperparameters for the model, evaluates it, and returns a score for the set of hyperparameters ( the value to maximize).

    This is the function that creates and trains a neural network with the given hyper-parameters, and then evaluates its
    performance on the validation-set. The function then returns the so-called fitness value (aka. objective value),
    which is the negative final_multimodal_Score on the validation-set. 
    It is negative because skopt performs minimization instead of maximization.

    It creates a model and train that with the parameters (lr and epsilon) passed
    as input. The trained model is used to make predictions on test and on syn
    and finally the predictions are used to compute the multimodal bias metric
    (value to maximize).
    """
    iteration = f_args.iteration
    syn_data = f_args.syn_data
	
    saving_folder = path_models + f'Iteration_{iteration}/'.format(iteration=iteration)
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # configure the data
    x_train, y_train, x_val, y_val, x_test, _ = preprocessing.elaborate_data_10fold(f_args.dataset, f_args.train_index,
                                                                                            f_args.test_index, iteration,
                                                                                            input_columns, label_column)


    # configure the model with specific hyperparameters
    model, _ = model_performances.get_trained_model(x_train, 
                            y_train, 
                            x_val, 
                            y_val,
                            input_shape=embed_size, 
                            activation_function=v_args.activation_function, 
                            neurons=v_args.neurons, 
                            dropout=v_args.dropout, 
                            lr=v_args.lr, 
                            epsilon=v_args.epsilon, 
                            epochs=epochs,
                            batch_size=batch_size)


    # save the model
    model.save(path_models + 'BO_model_unimodal_text_' + str(iteration))

    # make prediction on the test fold of Training data
    predict_values = model.predict(x_test, batch_size=batch_size)
    test_data = f_args.dataset.iloc[f_args.test_index, :].copy()
    test_data['pred_perc'] = predict_values

    csv_name = saving_folder + model_name + '_' + 'Train' + '_' + str(iteration) + '_' + str(int(time.time())) + '.csv'
    test_data.to_csv(csv_name, index=False, sep='\t')

    # make prediction on syn
    x_syn_BO, y_syn_BO = preprocessing.elaborate_input(syn_data, input_columns, label_column)

    predict_values = model.predict(x_syn_BO, batch_size=batch_size)
    syn_data['pred_perc'] = predict_values

    csv_name = saving_folder + 'unimodal_text_BO' + '_' + 'SYN' + '_' + str(iteration) + '_' + str(int(time.time())) + '.csv'
    syn_data.to_csv(csv_name, index=False, sep='\t')

    # BIAS:
    identity_terms_present = clear_identity_list([item for sublist in id_terms for item in sublist], syn_df)


    bias_metrics_text = model_performances.compute_bias_metrics_for_model(syn_data, identity_terms_present, 'pred_perc',
                                                                            label_column)

    overall_auc_metrics = adjusted_performances.calculate_overall_auc(test_data, 'pred_perc')

    final_unimodal_scores = adjusted_performances.get_final_uniimodal_metric_nan(bias_metrics_text,
                                                                                overall_auc_metrics, 'pred_perc')

    adjusted_performances.write_performance_on_file(saving_folder + 'results_{it}.txt'.format(it=iteration), iteration,
                                                    bias_metrics_text, overall_auc_metrics,
                                                    final_unimodal_scores)

    # Because we are interested in the HIGHEST classification
    # metric, we need to negate this number so it can be minimized
    return -final_unimodal_scores  # Metric to maximize

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

# new objective passed to the optimizer
# sub-version of the black_box_function (fitness) with blocked value that need to stay stable during optimization (e.g. iteration)
@use_named_args(dimensions=search_space)
def objective(lr, epsilon, neurons, activation_function, dropout):
    variable_args = VariableParams(lr, epsilon, neurons, activation_function, dropout)
    return fitness(variable_args, fixed_args) #fixed_args as global variable

#_________________________________________SYN preprocessing___________________________________
from ast import literal_eval
syn_df['tokens'] = syn_df['tokens'].apply(lambda x: literal_eval(x))

def contains_word(tokens, word):
    return any(word in tokens for word in [word])

for term in [item for sublist in id_terms for item in sublist]:
    syn_df[term]= ''
    syn_df[term]= syn_df['tokens'].apply(lambda x: contains_word(x, term))



print("train model on training data 10Fold for Test-SYN")
# ________________________________________train model on training data 10Fold________________________________________
iteration = 0
kf = KFold(n_splits=10, shuffle=False)

syn_10_df=pd.DataFrame()

res_syn = syn_df[['file_name', 'hard_label']].copy()
res_test = test_df[['file_name', 'hard_label']].copy()


for train_index, val_index in kf.split(train_df):
    preprocessing.set_seed(iteration)
    print('___ITERATION {it}___'.format(it=iteration))
    MODELNAME = MODELNAMES[iteration]
    # clear session
    tf.keras.backend.clear_session()

    #BO_test_syn, test_syn = syn_df[:len(syn_df)/2] #next(syn_folds)
    BO_syn_data = syn_df[:int(len(syn_df)/2)]
    syn_data = syn_df[int(len(syn_df)/2):]
    x_syn, y_syn = preprocessing.elaborate_input(syn_data, input_columns, label_column)

    fixed_args = FixedParams(iteration=iteration, dataset=train_df, syn_data=BO_syn_data, train_index=train_index, test_index=val_index)
    search_result = gp_minimize(func=objective,
                                dimensions=search_space,
                                acq_func='EI', # Expected Improvement.
                                n_calls=40,
                                x0=list(default_parameters.values()),
                                n_random_starts=5)
                        
    # TRAIN OF A NEW MODEL ON TRAINING DATA (9fold train-1fold val) TO COMPUTE FINAL METRICS
    x_train, y_train = preprocessing.elaborate_input(train_df.iloc[train_index, :], input_columns, label_column)
    x_val, y_val = preprocessing.elaborate_input(train_df.iloc[val_index, :], input_columns, label_column)

    optimal_config = dict(zip(list(default_parameters.keys()), search_result.x)) #uses the same keys in default_parameters

    model, _ = model_performances.get_trained_model(x_train, 
                                y_train, 
                                x_val, 
                                y_val,
                                input_shape=embed_size, 
                                activation_function=optimal_config['activation_function'], 
                                neurons=optimal_config['neurons'], 
                                dropout=optimal_config['dropout'], 
                                lr=optimal_config['lr'], 
                                epsilon=optimal_config['epsilon'], 
                                epochs=epochs,
                                batch_size=batch_size)

    # make prediction on test
    predict_values_test = model.predict(x_test, batch_size=batch_size)

    res_test[MODELNAME] = predict_values_test 
    res_test[['file_name', 'hard_label', MODELNAME]].to_csv(
        path_predictions + 'BO_Final_Test_' + str(iteration) + '_' + str(int(time.time())) + '.csv', index=False, sep='\t')

    # make prediction on syn
    predict_values_test = model.predict(x_syn, batch_size=batch_size)

    """ dataframe: for each model, three columns are added:
    - 'label_'+MODELNAME: real values
    - MODELNAME: predictions
    -'file_name_'+MODELNAME: ids of the memes

    """
    syn_10_df['label_'+MODELNAME]=list(syn_data[label_column].values)
    syn_10_df[MODELNAME]= predict_values_test
    syn_10_df['file_name_'+MODELNAME]=list(syn_data['file_name'].values)

    syn_data[MODELNAME] = predict_values_test 
    syn_data[['file_name', 'hard_label', MODELNAME]].to_csv(
        path_predictions + 'BO_Final_Syn_' + str(iteration) + '_' + str(int(time.time())) + '.csv', index=False, sep='\t')

    identity_terms_present = clear_identity_list([item for sublist in id_terms for item in sublist], syn_data)
    
    # BIAS:
    bias_metrics_text = model_performances.compute_bias_metrics_for_model(syn_data, identity_terms_present, MODELNAME,
                                                                        label_column)
    
    overall_auc_metrics = adjusted_performances.calculate_overall_auc(res_test, MODELNAME)

    final_unimodal_scores = adjusted_performances.get_final_unimodal_metric_nan(bias_metrics_text,
                                                                            overall_auc_metrics, MODELNAME)

    adjusted_performances.write_performance_on_file(results_txt_path, iteration, bias_metrics_text,
                                                overall_auc_metrics, final_unimodal_scores)
    iteration = iteration + 1

# Performance Syn 10Fold split
adjusted_performances.multimodal_bias_metrics_on_file_10Fold(results_txt_path, res_test, syn_10_df, syn_df, id_terms, MODELNAMES, label_column)

syn_10_df.to_csv(path_predictions + 'Mitigated_syn_10_df.csv', index=False, sep='\t')