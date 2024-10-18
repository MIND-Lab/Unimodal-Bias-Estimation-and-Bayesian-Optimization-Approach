import numpy as np
from sklearn import metrics
from Utils import model_performances

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

def get_final_uniimodal_metric_nan(bias_df_text, overall_auc_test, model_name):
    """compute AUC Final Meme _ a metric bias proposed to compute multimodal bias in memes
    it considers bias in text and in image """
    bias_score = compute_bias_score_nan(bias_df_text, model_name)
    return np.mean([overall_auc_test, bias_score])


# added to compute bias score on new synthetic set
def compute_bias_score_nan(bias_df_text, model_name):
    bias_score_text = np.nanmean([
        bias_df_text[model_name + '_subgroup_auc'],
        bias_df_text[model_name + '_bpsn_auc'],
        bias_df_text[model_name + '_bnsp_auc']
    ])
    return bias_score_text

def write_performance_on_file(filename, iteration, bias_metrics_text, overall_auc_metrics,
                              final_multimodal_scores):
    with open(filename, 'a+') as f:
        f.write('___ITERATION {it}___\n'.format(it=iteration))
        f.write('bias_metrics_text: {metrica} \n'.format(metrica=bias_metrics_text))
        f.write('overall_auc_metrics: {metrica} \n'.format(metrica=overall_auc_metrics))
        f.write('final_multimodal_scores:: {metrica} \n'.format(metrica=final_multimodal_scores))

def get_final_unimodal_metric_nan(bias_df_text, overall_auc_test, model_name):
    """compute AUC Final Meme _ a metric bias proposed to compute multimodal bias in memes
    it considers bias in text and in image """
    bias_score = compute_bias_score_nan(bias_df_text, model_name)
    return np.mean([overall_auc_test, bias_score])


def calculate_overall_auc(df, model_name):
    true_labels = df['label']
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

# Performances on Syn 10Fold
def multimodal_bias_metrics_on_file_10Fold(txt_path, test_df, syn_df, syn_df_complete, id_terms, model_names, label_column):
    """
    Compute metrics for bias-performance comparison and save them on file. Deals with nan values generated by synthetic dataset
    Allow to do deal with 10 fold execution on training data.

    :param txt_path: path to txt file to store results
    :param test_df: dataframe containing predictions on test data (columns should correspond to Modelnames)
    :param syn_df: dataframe containing predictions on synthetic data (columns should correspond to Modelnames and to
        identity elements)
    :param identity_terms: list of Identity Terms
    :param identity_tags: list of Identity Tags
    :param model_names: list of model names
    :return: /
    """

    final_multimodal_scores = {}
    bias_metrics_text = {}
    bias_value_unimodal_metrics = {}
    overall_auc_metrics = {}
    syn_data=syn_df.copy()
    for i in range(len(model_names)):
        syn_df=syn_data
        syn_df['file_name'] = syn_df['file_name_'+model_names[i]]
        # add information about the presence of identity terms
        syn_df=syn_df.merge(syn_df_complete.drop(columns=['soft_label_0', 'soft_label_1',
       'disagreement', 'sentences', 'tokens_lists', 'tokens_list',
       'lemmi_text', 'tokens', 'term_present']),
                        how='inner', on='file_name')
        syn_df['label'] = syn_df['label_'+model_names[i]]
        # select the subset of identity elements that are present
        identity_terms_present = clear_identity_list([item for sublist in id_terms for item in sublist], syn_df) #identity_element_presence(syn_data, Identity_Terms,label_column)
        
        bias_metrics_text[i] = model_performances.compute_bias_metrics_for_model(syn_df, identity_terms_present, model_names[i],
                                                              label_column)
        
        overall_auc_metrics[i] = calculate_overall_auc(test_df, model_names[i])
        final_multimodal_scores[i] = get_final_unimodal_metric_nan(bias_metrics_text[i],
                                                                 overall_auc_metrics[i],
                                                                 model_names[i])

        bias_value_unimodal_metrics[i] = np.nanmean([
                bias_metrics_text[i][model_names[i] + '_subgroup_auc'],
                bias_metrics_text[i][model_names[i] + '_bpsn_auc'],
                bias_metrics_text[i][model_names[i] + '_bnsp_auc']
            ])

    file = open(txt_path, "a+")
    file.write('\n Bias _ AUC_Final_Multimodal_SYN 10fold Split\n')
    file.write(
        'max overall auc: {max} \n'.format(max=max(zip(overall_auc_metrics.values(), overall_auc_metrics.keys()))))
    file.write('Overall AUC values: {values}\n'.format(values=overall_auc_metrics))
    file.write('average overall auc: {mean} \n'.format(mean=np.average(list(overall_auc_metrics.values()))))

    file.write('max multimodal bias: {max} \n'.format(
        max=max(zip(bias_value_unimodal_metrics.values(), bias_value_unimodal_metrics.keys()))))
    file.write('Multimodal bias values: {values}\n'.format(values=bias_value_unimodal_metrics))
    file.write(
        'average multimodal bias: {mean} \n'.format(mean=np.average(list(bias_value_unimodal_metrics.values()))))

    file.write('max AUC multimodal final: {max} \n'.format(
        max=max(zip(final_multimodal_scores.values(), final_multimodal_scores.keys()))))
    file.write('Multimodal final values: {values}\n'.format(values=final_multimodal_scores))
    file.write(
        'average AUC multimodal final: {mean} \n'.format(mean=np.average(list(final_multimodal_scores.values()))))
    file.close()