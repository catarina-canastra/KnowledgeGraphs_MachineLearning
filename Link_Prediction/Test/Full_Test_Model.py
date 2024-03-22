# -*- coding: utf-8 -*-
#
# Script: Full_Test_Model.py
# 
# Autor: Catarina Canastra and Catia Pesquita
#
# Use cases: use embeddings to get predictions with the different
#            link prediction embedding models of OpenKE library


####################################################################################
###                                                                              ###
###                             Libraries and Packages                           ###
###                                                                              ###
####################################################################################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd
import csv
import sys
import re
from OpenKE import config
from OpenKE import models

####################################################################################
###                                                                              ###
###                                  Load Model                                  ###
###                                                                              ###
####################################################################################

# output: model parameters and embeddings loaded/in RAM
def load_model(con, path_output, vector_size, model):

    # Set import files to OpenKE automatically loal models via tf.Saver()
    con.set_in_path(path_output)
    con.set_test_link_prediction(True)
    con.set_work_threads(8)
    con.set_dimension(vector_size)
    con.set_import_files(path_output + model + "/model.vec.tf")
    con.init()

    # Set model
    if model == 'TransE':
        con.set_model(models.TransE)
    if model == 'TransH':
        con.set_model(models.TransH)
    if model == 'TransR':
        con.set_model(models.TransR)
    if model == 'TransD':
        con.set_model(models.TransD)
    if model == 'distMult':
        con.set_model(models.DistMult)
    if model == 'RESCAL':
        con.set_model(models.RESCAL)
    if model == 'HOLE':
        con.set_model(models.HolE)
    if model == 'ComplEx':
        con.set_model(models.ComplEx)

    print('------------ Model Loaded ------------')

####################################################################################
###                                                                              ###
###            Get list of genes and diseases in which we are going              ###
###                      to predict the top K entities                           ###
###                                                                              ###
####################################################################################

# output: lists of genes, diseases and relations
def get_truth(test_file):

    '''
    This function is to receive a text file with all genes and associated diseases that correspond to the truth.
    Params:
    test_file - text file with all test triples in the format (ent1, ent2, rel)
    '''

    # read txt file
    file = open(test_file, 'r')
    file_correct = file.readlines()
    # discard firt line - is the number of triples
    file_correct = file_correct[1:]

    genes = []
    diseases = []
    relations = []

    for line in file_correct:
        gene, disease, relation = line[:-1].split('\t')
        genes.append(gene)
        diseases.append(disease)
        relations.append(relation)
    
    # close file
    file.close() # or just FILE

    return genes, diseases, relations
    
# output: mappings between the genes and diseases
# in practice: two dictionaries
# example: Gene A are associated with Disease 1 and Disease 2
def format_truth(genes, diseases):

    # Create a defaultdict where elements from 'genes' correspond to lists of elements from 'diseases'
    gene_to_diseases = defaultdict(list)
    for gene, disease in zip(genes, diseases):
        gene_to_diseases[str(gene)].append(str(disease))

    # Create a defaultdict where elements from 'diseases' correspond to lists of elements from 'genes'
    disease_to_genes = defaultdict(list)
    for gene, disease in zip(genes, diseases):
        disease_to_genes[str(disease)].append(str(gene))

    return gene_to_diseases, disease_to_genes

####################################################################################
###                                                                              ###
###               Filter out the results that aren't genes/diseases              ###
###                                                                              ###
####################################################################################

# output: gene_names, gene_ids
def get_genes(entity_file):

    '''
    This function is to get the entity file with entity names and IDs, and separate them in two lists.
    Params:
    entity_file - file with all entity names and corresponding IDs
    '''

    file = open(entity_file, 'r')
    # Ler logo desde a linha 1
    file_correct = file.readlines()[1:]

    #refs = []
    gene_names = []
    gene_ids = []

    for line in file_correct:
        # Tirar os ' que no fim complicam o prcesso de int 
        line = line.replace("'",'')
        # Rstrip retira logo os \n
        name, identifier = line.rstrip().split('\t')
        
            # ref = str(name)
        # Procurar por todas as entidades que terminam em numero a seguir a...
        # get all after "http://purl.obolibrary.org/obo/"
        filter_name = name.rpartition('/')[-1] # se o que está depois só contiver número, então é o ID de um gene e guardamos
        
        if filter_name.isdigit(): # se o que está depois só contiver número, então é o ID de um gene e guardamos
            gene_names.append(name)
            gene_ids.append(identifier)

    # close file
    file.close() # or just FILE

    print('------------ Genes extracted ------------')
    
    return gene_names, gene_ids

# output: disease_names, disease_ids
def get_diseases(entity_file):

    '''
    This function is to get the entity file with entity names and IDs, and separate them in two lists.
    Params:
    entity_file - file with all entity names and corresponding IDs
    '''

    file = open(entity_file, 'r')
    # Ler logo desde a linha 1
    file_correct = file.readlines()[1:]

    refs = []
    disease_names = []
    disease_ids = []

    for line in file_correct:
        # Tirar os ' que no fim complicam o prcesso de int 
        line = line.replace("'",'')
        # Rstrip retira logo os \n
        name, identifier = line.rstrip().split('\t')
        
        ref = str(name)
        # Procurar por todo o que é /C.....
        ref_object = re.search('/C(\d+[0-9])', ref)

        if ref_object:
            ref_id = ref_object.string.split('/C')[-1]
            if ref_id not in refs:
                refs.append(ref_id)
            else:
                refs = refs
            
            #####  ANALISE DOS NOMES #####
            #Pus de 1 para tirar o b e ficar só o url
            if name[1:] not in disease_names:
                disease_names.append(name[1:])
            else:
                disease_names = disease_names

            ##### ANALISE DOS IDS #####
            if identifier not in disease_ids:
                disease_ids.append(identifier)
            else:
                disease_ids = disease_ids

    # close file
    file.close() # or just FILE

    print('------------ Diseases extracted ------------')

    return disease_names, disease_ids

# output: topk valid predictions
def filter_out_results(list_ids, results, k):

    '''
    This function is to filter out the predictions that are not diseases (Cxxxxxx).
    Params:
    disease_ids - a list with all disease IDs
    df - a dataframe with the predictions made
    '''

    #print('Some examples of DF:', df.head())
 
    #predictions = results # df['Predictions'].to_list()
    #predicted = len(predictions)

    valid_results = []

    for pred in results:
        id = str(pred)
        if id in list_ids:
            valid_results.append(id)
    
    topk = valid_results[:k]

    #valid = len(valid_results)

    #print('Some examples in VALID RESULTS:', valid_results[:5])
    print('------------ Results filtered ------------')
    #print('The model made', predicted ,'predictions, of which', valid ,'are valid.')

    return topk

# output: dataframe with the genes, predictions and ranks
def format_predictions(topk, type_entity, entity):

    '''
    This function is to get both the truth and the predictions and create two datasets.
    The datasets contain several columns: Gene, Disease, Relevancy (CCCC) and Rank (except the truth dataset).
    Params:
    genes - a list of genes that we want to predict the associated disease
    diseases - a list of diseases trully associated with the genes mentioned above
    valid_results - a dataframe with valid predictions
    '''

    # when we are predicting the candidate diseases for a gene
    if type_entity == 'Gene':
        # format the dataset with the predicted results - K per gene (later we have each result in a line)
        data_result = {
            "Disease": topk
        }

        prediction = pd.DataFrame(data_result)
        # specifies where we want the new column (0 - to the left)
        prediction.insert(0, 'Gene', str(entity))
        prediction['Rank'] = prediction.index + 1
    
    # when we are predicting the candidate genes for a disease
    else:
        # format the dataset with the predicted results - K per gene (later we have each result in a line)
        data_result = {
            "Gene": topk
        }

        prediction = pd.DataFrame(data_result)
        # specifies where we want the new column (0 - to the left)
        prediction.insert(1, 'Disease', str(entity))
        prediction['Rank'] = prediction.index + 1

    #print('------------ Formatted predictions ------------')
    
    return prediction

####################################################################################
###                                                                              ###
###             Get ranks of the righ predictions inside topk                    ###
###                                                                              ###
####################################################################################

# output: ranks of the predictions trully associated with the entity
def get_ranks(dic_truth, entity, type_entity, prediction):

    # initiate empty lists
    ranks = []

    # get the entities trully associated with the entity we got predictions
    trully_entities = dic_truth[str(entity)]
    # (value for key, value in dic_truth.items() if key == str(entity))
    print(trully_entities)

    if type_entity == 'Gene':
        for associate in trully_entities: # eu podia meter isto antes do if
            print(type(associate)) # eu podia meter isto antes do if
            condition = prediction['Disease'] == associate
            if condition.any():
               ranks.append(prediction.loc[condition, 'Rank'].values[0])
            else:
               ranks.append(0)

    else:
        for associate in trully_entities: # eu podia meter isto antes do if
            print(type(associate)) # eu podia meter isto antes do if
            condition = prediction['Gene'] == associate
            if condition.any():
               ranks.append(prediction.loc[condition, 'Rank'].values[0])
            else:
               ranks.append(0)
    
    print(ranks)

    return ranks

####################################################################################
###                                                                              ###
###                            Model CSV file                                    ###
###                                                                              ###
####################################################################################

# output: plus one line in a CSV with the entity type (gene or disease), the predicted gene/disease,
#  and the top k predicted associated entities
def save_topk(csv_top, k, type_entity, entity, topk):

    # Check if the file exists; if it doesn't, write the header row
    file_exists = True
    try:
        with open(csv_top, 'r') as file:
            if not any(line.strip() for line in file):
                file_exists = False
    except FileNotFoundError:
        file_exists = False

    # Open the CSV file for writing in append mode
    with open(csv_top, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # If the file doesn't exist, write the header row
        if not file_exists:
            csv_writer.writerow(["Entity_Type", "Entity", "Top" + str(k)])

        # Convert the topk list to a comma-separated string
        topk_str = ', '.join(map(str, topk))

        # Write the entity and its top 100 values as a row in the CSV file
        if type_entity == 'Gene':
            csv_writer.writerow(['Gene', entity, topk_str])
        else:
            csv_writer.writerow(['Disease', entity, topk_str])

    print(f"Data has been appended to {csv_top}")

    # Close the CSV file
    csv_file.close()

# output: plus one line in a CSV with the entity type (gene or disease), the predicted gene/disease, 
# and the ranks of the trully associated predictions
def save_ranks(csv_ranks, type_entity, entity, ranks):

    # Check if the file exists; if it doesn't, write the header row
    file_exists = True
    try:
        with open(csv_ranks, 'r') as file:
            if not any(line.strip() for line in file):
                file_exists = False
    except FileNotFoundError:
        file_exists = False

    # Open the CSV file for writing in append mode
    with open(csv_ranks, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL) # to have quotes also when we have one rank

        # If the file doesn't exist, write the header row
        if not file_exists:
            csv_writer.writerow(["Entity_Type", "Entity", "Ranks"])
        
        # Convert the ranks list to a comma-separated string
        ranks_str = ', '.join(map(str, ranks))

        # Write the entity and its top 100 values as a row in the CSV file
        if type_entity == 'Gene':
            csv_writer.writerow(['Gene', entity, ranks_str]) # TEM DE SER UMA LISTA 
        else:
            csv_writer.writerow(['Disease', entity, ranks_str])

    print(f"Data has been appended to {csv_ranks}")

    # Close the CSV file
    csv_file.close()

####################################################################################
###                                                                              ###
###                             Put all to run                                   ###
###                                                                              ###
####################################################################################

def run_all(entity_file, path_output, vector_size, model, test_file, type, entity, id_relation, k):

    # get all gene ids
    gene_names, gene_ids = get_genes(entity_file)
    del gene_names
    # get all disease ids
    disease_names, disease_ids = get_diseases(entity_file)
    del disease_names

    # call Config class
    con = config.Config()

    # load model
    load_model(con, path_output, vector_size, model)

    # Define CSV file names
    csv_top = model + '_Top' + str(k) + '.csv'
    csv_ranks = model + '_Ranks.csv'

    # get the truth results (diseases associated with the genes)
    genes, diseases, relations = get_truth(test_file)
    del relations

    # format the truth results to have a mapping between each entity and their corresponding associated entities
    gene_to_diseases, disease_to_genes = format_truth(genes, diseases)
    del genes, diseases

    # iterate over the list of genes to predict the top K entities for each gene
    #for gene in unique_genes_list:
    if type == 'Gene':
        # get predictions
        results = con.predict_tail_entity(entity, id_relation)
        # filter out the results that are not OBO diseases and get top k results
        topk = filter_out_results(disease_ids, results, k)
        del results
        # save topk in a csv
        save_topk(csv_top, k, type, entity, topk)
        # format datasets with the truth results and the predictions
        prediction = format_predictions(topk, type, entity)
        del topk
        # get the ranks of the right answers
        ranks = get_ranks(gene_to_diseases, entity, type, prediction)
        del prediction
        # Save ranks of relevancy = 1 in a CSV file
        save_ranks(csv_ranks, type, entity, ranks)
        del ranks
    
    #del gene_to_diseases

    # iterate over the list of diseases to predict the top K entities for each disease
    #for disease in unique_diseases_list:
    else:
        # get predictions
        results = con.predict_head_entity(entity, id_relation)
        # filter out the results that are not OBO diseases and get top k results
        topk = filter_out_results(gene_ids, results, k)
        del results
        # save topk in a csv
        save_topk(csv_top, k, type, entity, topk)
        # format datasets with the truth results and the predictions
        prediction = format_predictions(topk, type, entity)
        del topk
        # get the ranks of the right answers
        ranks = get_ranks(disease_to_genes, entity, type, prediction)
        del prediction
        # Save ranks of relevancy = 1 in a CSV file
        save_ranks(csv_ranks, type, entity, ranks)
        del ranks

    del gene_to_diseases, disease_to_genes

    print('----------------- End ------------------')

####################################################################################
###                                                                              ###
###                                    Main                                      ###
###                                                                              ###
####################################################################################

def main():
    parser = argparse.ArgumentParser(description="Script where you need to specify which algoritm you want to use, the ID of the relation hasAssociation, the type of the entity, the value of the entity and how many candidate entities you want to save.")
    parser.add_argument("--algorithm", choices=["TransE", "TransD", "TransH", "TransR", "distMult", "RESCAL", "HOLE", "ComplEx"], required=True, help="Specify the algorithm")
    parser.add_argument("--entity_type", choices=["Gene", "Disease"], required=True, help="Specify the type of the entity")
    parser.add_argument("--entity", type=str, required=True, help="Pass the entity")
    parser.add_argument("--relation_id", type=int, required=True, help="Specify the relation ID")
    parser.add_argument("--top_k", type=int, required=True, help="Specify the top K")

    args = parser.parse_args()
    
    run_all(entity_file, path_output, vector_size, args.algorithm, test_file,args.entity_type, args.entity, args.relation_id, args.top_k)

####################################################################################
###                                                                              ###
###                               Calling Functions                              ###
###                                                                              ###
####################################################################################

# Define output path
path_output =  './'

# Example of running
entity_file = 'entity2id.txt'
test_file = 'test2id.txt'
#id_relation = 32
#k = 1000
vector_size = 200

# Define model
# Options: 'TransE', 'TransD', 'TransH', 'TransR', 'distMult', 'RESCAL', 'HOLE', 'ComplEx'
#model = 'TransD'

#run_all(entity_file, path_output, vector_size, model, test_file, id_relation, k)
# SUBSTITUTED BY
if __name__ == "__main__":
    main()