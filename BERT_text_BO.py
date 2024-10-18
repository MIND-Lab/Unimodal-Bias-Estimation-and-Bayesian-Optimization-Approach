from sklearn.model_selection import KFold
import time
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from collections import namedtuple
import torch.cuda
import torch
import random
import numpy as np
import os
from datasets import Dataset
from transformers import BertTokenizer,AutoTokenizer,BertForSequenceClassification, TrainingArguments, Trainer
import evaluate
import pandas as pd
import gc
from Utils import  bert_basics,preprocessing, model_performances, adjusted_performances
from sklearn.model_selection import KFold

metric = evaluate.load("accuracy")
##Set random values
seed_val = 222
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed_val)

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    #device = torch.device('cuda:1')
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#____________________________functions__________________________________

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

def batch_predict(model, texts, tokenizer, batch_size=16, device="cuda"):
    model = model.to(device)
    all_predictions = []
    all_probabilities = []
    
    # Iterate over the texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # Preprocess batch of texts
        inputs = preprocess_text(batch_texts, tokenizer)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        # Make predictions
        predicted_classes, probabilities = predict_text(model, inputs)
        all_predictions.append(predicted_classes.cpu())
        all_probabilities.append(probabilities.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions)
    all_probabilities = torch.cat(all_probabilities)
    
    return all_predictions, all_probabilities

dataset_name = 'ConvAbuse'
id_terms = [['asshole', 'pussy', 'fuck', 'stupid', 'sex', 'idiot', 'kill', 'hate', 'bitch', 'slut'], ['mention', 'dead', 'english', 'nope', 'hungry', 'ever', 'old', 'first', 'meet', 'keep']]

#dataset_name = "HS-Brexit"
#id_terms =[['paki', 'deport', '#trump2016', 'islam', 'obama', 'muslim', 'world', 'illegal', 'invasion', 'white'], ['blame', 'cause', 'terrorism', 'might', 'another', 'attack', 'watch', '#remain', 'bad', '#brexitvote']]

##dataset_name = 'MD-Agreement'
#id_terms = [['shitty', 'cunt', 'fuck', 'stfu', 'hitler', 'bullshit', 'fuckin', 'niggas', 'asshole', 'pedophile'], ['insurance', 'ballot', 'anger', 'difficult', 'wall', 'cancel', 'across', 'replace', 'company', '#anonymous']]

#dataset_name = "ArMIS"
#id_terms = [['الفسويات', 'وقحات', 'قذرات', 'النار', 'متسلطات', 'النسويات', 'ايش', 'ههه', 'رخيصات', 'وهي'], ['نسويه', 'تقول', 'يقول', 'كانوا', 'لأن', 'العنف', 'النسويه', 'علي', 'المراه', 'يأخذ']]

# ________________________________________Utils ___________________________________________________
if not os.path.exists('./'+dataset_name+'/BERT/predictions'):
    os.makedirs('./'+dataset_name+'/BERT/predictions')

if not os.path.exists('./'+dataset_name+'/BERT/performances'):
    os.makedirs('./'+dataset_name+'/BERT/performances')

path_models = './'+dataset_name+'/BERT/models'
file_out = './'+dataset_name+'/BERT/performances/BERT_BO_results_10Fold.txt'
predictions_csv_path = './BERT/'+dataset_name+'/predictions/BERT_BO_pred_10Fold.csv'

file = open(file_out, 'a+')
file.truncate(0)  # erase file content
file.close()


label_column = "hard_label"
input_columns = ['original_text']
threshold = 0.5
embed_size = 512  # 512-length array with Universal Sentence Encoder algorithm
batch_size = 16

# ________________________________________Load Data ___________________________________________________
print("Loading data...")
folder = "./Data/"
# ________________________________________load training data ___________________________________________________

train_df = pd.read_json(folder + dataset_name + "_dataset/" + dataset_name + "_train.json", orient='index')
train_df = preprocessing.get_dataset_labels(train_df)

train_df['file_name'] =train_df.index
train_df = train_df.rename(columns={input_columns[0]:'text',label_column: 'label' })


# ________________________________________Utils ___________________________________________________
path_predictions = './'+dataset_name+'/BERT/predictions/'
path_performances = './'+dataset_name+'/BERT/performances/'
file_out = path_performances + 'text_results_10Fold_BO.txt'
predictions_csv_path = path_predictions + 'text_pred_10Fold_BO.csv'

for path in [path_models, path_predictions, path_performances]:
    if not os.path.exists(path):
        os.makedirs(path)

file = open(file_out, 'a+')
file.truncate(0)  # erase file content
file.close()


# ________________________________________load Test + SYN data ___________________________________________________
print("loading test+syn...")
# Load Test and preprocessing
test_df =  pd.read_json(folder +"/"+dataset_name+"_dataset/"+dataset_name+"_test.json", orient='index')
test_df = preprocessing.get_dataset_labels(test_df)

test_df = test_df.rename(columns={input_columns[0]:'text',label_column: 'label' })
test_df['file_name']=test_df.index

syn_df = pd.read_csv('Syn_df/'+dataset_name+ '_syn.csv', sep='\t')

syn_df = syn_df.rename(columns={input_columns[0]:'text',label_column: 'label' })
syn_df['file_name'] = syn_df.index 

res_syn = syn_df[['file_name', 'label']].copy()
res_test = test_df[['file_name', 'label']].copy()

input_columns = ['text']
label_column = 'label'

x_test, y_test = bert_basics.elaborate_input(test_df, input_columns, label_column)
syn_test = syn_df[int(len(syn_df)/2):]
syn_data = syn_test.copy()
x_syn, y_syn = bert_basics.elaborate_input(syn_test, input_columns, label_column)

#_________________________Models on Traing-Test Data _____________________________________
MODELNAMES = ['BERT_text_BO_v{}'.format(i) for i in range(10)]
path_models = './'+dataset_name+"/Mitigated/models/"
path_predictions= './'+dataset_name+"/Mitigated/predictions/"
path_performances = './'+dataset_name+"/Mitigated/performances"
model_name = 'BERT_text_BO'

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

default_parameters={"lr":5e-5, #0.00001,
        "epsilon": 1e-8 #e-7
    }

# distinguish variable parameter from fixed parameters in 10Fold execution
VariableParams = namedtuple('VariableParams', 'lr epsilon')
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
    print("fitness function...")
    iteration = f_args.iteration
    syn_data = f_args.syn_data
	
    
    saving_folder = path_models + f'Iteration_{iteration}/'.format(iteration=iteration)
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    
    # configure the data
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.elaborate_data_10fold_bert(f_args.dataset, f_args.train_index,
                                                                                            f_args.test_index, iteration,
                                                                                            input_columns, label_column)


    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=len(train_df[label_column].unique()))

    # Tokenize the datasets
    train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(x_val, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=512)

    # Create torch datasets
    train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 
                                       'attention_mask': train_encodings['attention_mask'], 
                                       'labels': y_train})
    val_dataset = Dataset.from_dict({'input_ids': val_encodings['input_ids'], 
                                     'attention_mask': val_encodings['attention_mask'], 
                                     'labels': y_val})
    """
    test_dataset = Dataset.from_dict({'input_ids': test_encodings['input_ids'], 
                                      'attention_mask': test_encodings['attention_mask'], 
                                      'labels': y_test})
    """
    training_args = TrainingArguments(output_dir=path_predictions+str(iteration), 
                                      save_strategy="no",
                                      evaluation_strategy="epoch", 
                                      num_train_epochs=3,
                                      learning_rate=v_args.lr,
                                      adam_epsilon = v_args.epsilon,
                                      )


    # Initialize Trainer
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        compute_metrics=compute_metrics      # custom metrics function
    )

    # Train the model
    trainer.train()

    # save the model
    #model.save(path_models + 'BO_model_unimodal_text_' + str(iteration))
    """
    print("model saving...")
    saving_path = './'+dataset_name+'/BERT/models/BO_step'+str(iteration)+'/model.pth'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    #torch.save(model.state_dict(),saving_path )
    model.save_pretrained(saving_path)
    """
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])

    predict_values, _ = batch_predict(model, x_test, tokenizer, batch_size=16, device=device)
    
    test_data = f_args.dataset.iloc[f_args.test_index, :]
    test_data['pred_perc'] = predict_values.cpu()

    #csv_name = saving_folder + model_name + '_' + 'Train' + '_' + str(iteration) + '_' + str(int(time.time())) + '.csv'
    #test_data.to_csv(csv_name, index=False, sep='\t')

    # make prediction on syn
    x_syn_BO, y_syn_BO = bert_basics.elaborate_input(syn_data, input_columns, label_column)

    # make prediction on syn
    predicted_classes_s, _ = batch_predict(model, x_syn_BO, tokenizer, batch_size=16, device=device)
    

    syn_data['pred_perc'] = predicted_classes_s.cpu()

    #csv_name = saving_folder + 'unimodal_text_BO' + '_' + 'SYN' + '_' + str(iteration) + '_' + str(int(time.time())) + '.csv'
    #syn_data.to_csv(csv_name, index=False, sep='\t')

    # BIAS:
    identity_terms_present = clear_identity_list([item for sublist in id_terms for item in sublist], syn_df) #model_performances.identity_element_presence(syn_data, Identity_Terms, label_column)


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
        if len(df.loc[df[tag]==True,'label'].value_counts())<2:
            to_remove.append(tag)
    for tag in to_remove:
        identity_list.remove(tag)
    return identity_list

# new objective passed to the optimizer
# sub-version of the black_box_function (fitness) with blocked value that need to stay stable during optimization (e.g. iteration)
@use_named_args(dimensions=search_space)
def objective(lr, epsilon):
    variable_args = VariableParams(lr, epsilon)
    return fitness(variable_args, fixed_args) #fixed_args as global variable

#_________________________________________SYN preprocessing___________________________________
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
        if len(df.loc[df[tag]==True,'label'].value_counts())<2:
            to_remove.append(tag)
    for tag in to_remove:
        identity_list.remove(tag)
    return identity_list


print("train model on training data 10Fold for Test-SYN")
# ________________________________________train model on training data 10Fold________________________________________

# split train in 10 folds
train_df = train_df.reset_index(drop=True)

iteration = 0
real_values = np.array([])
predict_values = np.array([])
ids = np.array([])
train_df = train_df.fillna('')

pred_syn=[]
kf = KFold(n_splits=10, shuffle=False)

syn_10_df=pd.DataFrame()

res_syn = syn_test[['file_name', 'label']].copy()
res_test = test_df[['file_name', 'label']].copy()

syn_data = syn_df[int(len(syn_df)/2):]

for train_index, val_index in kf.split(train_df):
    preprocessing.set_seed(iteration)
    print('___ITERATION {it}___'.format(it=iteration))
    MODELNAME = MODELNAMES[iteration]

    BO_syn_data = syn_df[:int(len(syn_df)/2)]

    fixed_args = FixedParams(iteration=iteration, dataset=train_df, syn_data=BO_syn_data, train_index=train_index, test_index=val_index)
    search_result = gp_minimize(func=objective,
                                dimensions=search_space,
                                acq_func='EI', # Expected Improvement.
                                n_calls=40,
                                x0=list(default_parameters.values()),
                                n_random_starts=5)
                        
    gc.collect()
    torch.cuda.empty_cache()

    
    # TRAIN OF A NEW MODEL ON TRAINING DATA (9fold train-1fold val) TO COMPUTE FINAL METRICS
    x_train, y_train = bert_basics.elaborate_input(train_df.iloc[train_index, :], input_columns, label_column)
    x_val, y_val = bert_basics.elaborate_input(train_df.iloc[val_index, :], input_columns, label_column)

    optimal_config = dict(zip(list(default_parameters.keys()), search_result.x)) #uses the same keys in default_parameters

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=len(train_df[label_column].unique()))

    # Tokenize the datasets
    train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(x_val, truncation=True, padding=True, max_length=512)
    
    # Create torch datasets
    train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 
                                       'attention_mask': train_encodings['attention_mask'], 
                                       'labels': y_train})
    val_dataset = Dataset.from_dict({'input_ids': val_encodings['input_ids'], 
                                     'attention_mask': val_encodings['attention_mask'], 
                                     'labels': y_val})
    

    training_args = TrainingArguments(output_dir=path_predictions+str(iteration)+'_test', 
                                      save_strategy="no",
                                      evaluation_strategy="epoch", 
                                      learning_rate=optimal_config['lr'],
                                      adam_epsilon = optimal_config['epsilon'],
                                      num_train_epochs=3) #5)


    # Initialize Trainer
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        compute_metrics=compute_metrics,      # custom metrics function
    )

    # Train the model
    trainer.train()

    gc.collect()
    torch.cuda.empty_cache()

    # WRITE optimal_config on File
    file = open("./"+ dataset_name + "optimal_config.txt", "a+")
    file.write(str(optimal_config))
    file.write("\n")


    """
    saving_path = './BERT/models/'+str(iteration)+'/model.pth'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    #torch.save(model.state_dict(),saving_path )
    model.save_pretrained(saving_path)
    """

    """ TRAIN + TEST e SYN """

    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])

    predicted_classes, probabilities = batch_predict(model, x_test, tokenizer, batch_size=16, device=device)
    

    probs_0 = []
    probs_1 = []
    for item in probabilities:
        probs_0.append(item[0].item())
        probs_1.append(item[1].item())

    predict_values = np.append(predict_values, predicted_classes.cpu())
    real_values = np.append(real_values, y_test)
    ids = np.append(ids, test_df['file_name'].tolist())


    res_test[MODELNAME] = predicted_classes.cpu() #pd.DataFrame(predict_values)[0]
    res_test[MODELNAME+'_prob'] = probs_1


    res_test[['file_name', 'label', MODELNAME]].to_csv(
        path_predictions + 'BO_Final_Test_' + str(iteration) + '_' + str(int(time.time())) + '.csv', index=False, sep='\t')

    # make prediction on syn
    predicted_classes_s, probabilities_s = batch_predict(model, x_syn, tokenizer, batch_size=16, device=device)
    
    # performances on splitted Syn
    syn_10_df['label_'+MODELNAME]=list(syn_test[label_column].values)
    syn_10_df[MODELNAME]= predicted_classes_s.cpu()
    syn_10_df['file_name_'+MODELNAME]=list(syn_test['file_name'].values)

    syn_data[MODELNAME] = predicted_classes_s.cpu()

    syn_data[['file_name', 'label', MODELNAME]].to_csv(
        path_predictions + 'BO_Final_Syn_' + str(iteration) + '_' + str(int(time.time())) + '.csv', index=False, sep='\t')


    identity_terms_present = clear_identity_list([item for sublist in id_terms for item in sublist], syn_data) #identity_element_presence(syn_data, Identity_Terms,label_column)
    
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