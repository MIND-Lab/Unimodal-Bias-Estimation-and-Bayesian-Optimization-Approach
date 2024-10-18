import torch.cuda
import torch
import random
import numpy as np
import os
from datasets import Dataset
from transformers import BertTokenizer,BertForSequenceClassification, TrainingArguments, Trainer
import evaluate

import pandas as pd
from Utils import  bert_basics,preprocessing, model_performances
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

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

dataset_name = 'ConvAbuse'
id_terms = [['asshole', 'pussy', 'fuck', 'stupid', 'sex', 'idiot', 'kill', 'hate', 'bitch', 'slut'], ['mention', 'dead', 'english', 'nope', 'hungry', 'ever', 'old', 'first', 'meet', 'keep']]

#dataset_name = 'HS-Brexit'
#id_terms = [['paki', 'deport', '#trump2016', 'islam', 'obama', 'muslim', 'world', 'illegal', 'invasion', 'white'], ['blame', 'cause', 'terrorism', 'might', 'another', 'attack', 'watch', '#remain', 'bad', '#brexitvote']]

#dataset_name = 'MD-Agreement'
#id_terms = [['shitty', 'cunt', 'fuck', 'stfu', 'hitler', 'bullshit', 'fuckin', 'niggas', 'asshole', 'pedophile'], ['insurance', 'ballot', 'anger', 'difficult', 'wall', 'cancel', 'across', 'replace', 'company', '#anonymous']]

#dataset_name = 'ArMIS'
#id_terms = [['الفسويات', 'وقحات', 'قذرات', 'النار', 'متسلطات', 'النسويات', 'ايش', 'ههه', 'رخيصات', 'وهي'], ['نسويه', 'تقول', 'يقول', 'كانوا', 'لأن', 'العنف', 'النسويه', 'علي', 'المراه', 'يأخذ']]
# ________________________________________Utils ___________________________________________________
if not os.path.exists('./'+dataset_name+'/BERT/predictions'):
    os.makedirs('./'+dataset_name+'/BERT/predictions')

if not os.path.exists('./'+dataset_name+'/BERT/performances'):
    os.makedirs('./'+dataset_name+'/BERT/performances')

path_models = './'+dataset_name+'/BERT/models'
file_out = './'+dataset_name+'/BERT/performances/BERT_results_10Fold.txt'
predictions_csv_path = './BERT/'+dataset_name+'/predictions/BERT_pred_10Fold.csv'

file = open(file_out, 'a+')
file.truncate(0)  # erase file content
file.close()


label_column = "hard_label"
input_columns = ['original_text']
threshold = 0.5
embed_size = 512  # 512-length array with Universal Sentence Encoder algorithm

# ________________________________________Load Data ___________________________________________________
print("Loading data...")
from Utils import preprocessing as pr
folder = "./Data/"
# ________________________________________load training data ___________________________________________________

train_df = pd.read_json(folder + dataset_name + "_dataset/" + dataset_name + "_train.json", orient='index')
train_df = pr.get_dataset_labels(train_df)

train_df['file_name'] =train_df.index
train_df = train_df.rename(columns={input_columns[0]:'text',label_column: 'label' })


# ________________________________________load Test + SYN data ___________________________________________________
# Load Test and preprocessing
test_df =  pd.read_json(folder +"/"+dataset_name+"_dataset/"+dataset_name+"_test.json", orient='index')
test_df = pr.get_dataset_labels(test_df)

test_df = test_df.rename(columns={input_columns[0]:'text',label_column: 'label' })
test_df['file_name']=test_df.index


syn_df = pd.read_csv('Syn_df/'+dataset_name+ '_syn.csv', sep='\t')
# ilsolo la seconda metà di SYN
syn_df = syn_df[int(len(syn_df)/2):]
syn_df = syn_df.rename(columns={input_columns[0]:'text',label_column: 'label' })
syn_df['file_name'] = syn_df.index 

res_syn = syn_df[['file_name', 'label']].copy()
res_test = test_df[['file_name', 'label']].copy()

input_columns = ['text']
label_column = 'label'

x_test, y_test = bert_basics.elaborate_input(test_df, input_columns, label_column)
x_syn, y_syn = bert_basics.elaborate_input(syn_df, input_columns, label_column)

#_________________________Models on Traing-Test Data _____________________________________

MODELNAMES = ['BERT_text_v{}'.format(i) for i in range(10)]
path_models = './'+dataset_name+'/BERT/models/Bias/'
model_name = 'BERT_text'

if not os.path.exists(path_models):
    os.makedirs(path_models)

print("train model on training data 10Fold for Test-SYN")

path_predictions = './'+dataset_name+'/BERT/predictions/'
MODELNAMES = ['BERT_text_v{}'.format(i) for i in range(10)]
path_models = './'+dataset_name+'/BERT/models/Bias/'
path_results_test = './'+dataset_name+'/BERT/predictions/predictions_test/'
path_results_syn = './'+dataset_name+'/BERT/predictions/predictions_syn/'
path_performances = './'+dataset_name+'/BERT/performances'
model_name = 'BERT_text'

file_out_test = "./"+dataset_name+"/BERT/performances/text_Results_Test.txt"
file_out_syn = "./"+dataset_name+"/BERT/performances/text_Results_Syn.txt"
file_out_bias = "./"+dataset_name+"/BERT/performances/text_Results_Bias.txt"
for path in [path_results_test, path_results_syn, path_performances]:
    if not os.path.exists(path):
        os.makedirs(path)

for file_name in [file_out_test, file_out_syn, file_out_bias]:
    file = open(file_name, 'a+')
    file.truncate(0)  # erase file content
    file.close()
# ________________________________________train model on training data 10Fold________________________________________

train_df = train_df.reset_index(drop=True)

kf = KFold(n_splits=10, shuffle=False)

iteration = 0
real_values = np.array([])
predict_values = np.array([])
ids = np.array([])

train_df = train_df.fillna('')

pred_syn=[]
syn_10_df=pd.DataFrame()

for train_index, val_index in kf.split(train_df):  # split into train and test
    
    preprocessing.set_seed(iteration)
    MODELNAME = MODELNAMES[iteration]

    x_train, y_train = bert_basics.elaborate_input(train_df.iloc[train_index, :], input_columns, label_column)
    x_val, y_val = bert_basics.elaborate_input(train_df.iloc[val_index, :], input_columns, label_column)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=len(train_df[label_column].unique()))

    # Tokenize the datasets
    train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=embed_size)
    val_encodings = tokenizer(x_val, truncation=True, padding=True, max_length=embed_size)
    
    # Create torch datasets
    train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 
                                       'attention_mask': train_encodings['attention_mask'], 
                                       'labels': y_train})
    val_dataset = Dataset.from_dict({'input_ids': val_encodings['input_ids'], 
                                     'attention_mask': val_encodings['attention_mask'], 
                                     'labels': y_val})
    

    training_args = TrainingArguments(output_dir=path_predictions+str(iteration)+'_test', 
                                      evaluation_strategy="epoch", 
                                      num_train_epochs=3) #5)


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

    print("model saving...")
    saving_path = './BERT/models/'+str(iteration)+'/model.pth'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    #torch.save(model.state_dict(),saving_path )
    model.save_pretrained(saving_path)

    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    
    print("making predictions...")
    inputs = preprocess_text(x_test, tokenizer)
    predicted_classes, probabilities = predict_text(model.to(device), inputs.to(device))
    
    probs_0 = []
    probs_1 = []
    for item in probabilities:
        probs_0.append(item[0].item())
        probs_1.append(item[1].item())

    predict_values = np.append(predict_values, predicted_classes.cpu())
    real_values = np.append(real_values, y_test)
    ids = np.append(ids, test_df['file_name'].tolist())

    res_test[MODELNAME] = predicted_classes.cpu()
    res_test[MODELNAME+'_prob'] = probs_1

    # make prediction on syn
    inputs_s = preprocess_text(x_syn, tokenizer)
    predicted_classes_s, probabilities_s = predict_text(model.to(device), inputs_s.to(device))
    res_syn[MODELNAME] = predicted_classes_s.cpu()

    # performances on splitted Syn
    syn_10_df['label_'+MODELNAME]=list(res_syn[label_column].values)
    syn_10_df[MODELNAME]= list(res_syn[MODELNAME].values)
    syn_10_df['file_name_'+MODELNAME]=list(res_syn['file_name'].values)
    print("iteration " + str(iteration)+ " done!")
    iteration = iteration + 1
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
        if len(df.loc[df[tag]==True,'label'].value_counts())<2:
            to_remove.append(tag)
    for tag in to_remove:
        identity_list.remove(tag)
    return identity_list

from Utils2 import load_data
Identity_Terms = clear_identity_list([item for sublist in id_terms for item in sublist], syn_df)

res_syn = res_syn.merge(syn_df.drop(columns=['text', 'label', 'soft_label_0', 'soft_label_1',
       'disagreement', 'sentences', 'tokens_lists', 'tokens_list',
       'lemmi_text', 'tokens', 'term_present']),
                        how='inner', on='file_name')

res_syn['misogynous'] = res_syn['label']
res_test['misogynous'] = res_test['label']

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