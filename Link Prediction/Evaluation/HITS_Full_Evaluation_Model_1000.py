# -*- coding: utf-8 -*-
#
# Script: Full_Test_Model.py
# 
# Autor: Catarina Canastra and Catia Pesquita
#
# Use cases: evaluate KGE model's performance, particularly OpenKE models (with Tensorflow)


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
import numpy as np
import json
import pandas as pd
import csv
import sys
import re
import collections

####################################################################################
###                                                                              ###
###                                  Load Ranks                                  ###
###                                                                              ###
####################################################################################

'''
def load_ranks(path_ranks):

    # initate empty dictionary to store the entity (key), the type of entity (value[0])
    # and the ranks of the associated entities (value[1])
    entity_ranks = {}

    # load data from the CSV file
    with open(path_ranks, 'r') as file:
        reader = csv.DictReader(file, quoting=csv.QUOTE_ALL)
        next(reader)
        for row in reader:
            type_entity = row['Entity_Type']
            entity = row['Entity']
            ranks = row['Ranks'].split(',')
            entity_ranks[entity] = (type_entity, ranks)

    # prints
    for entity, (type_entity, ranks) in entity_ranks.items():
        print(f"Entity: {entity}, Type of entity: {type_entity}, List: {', '.join(ranks)}")
    
    return entity_ranks
'''

'''
def read_csv_to_dict(file_path):
    data_dict = {}  # Initialize an empty dictionary to store the data

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Assuming 'Entity' column uniquely identifies each row
            entity_id = row['Entity']  # Modify this key according to your CSV structure
            data_dict[entity_id] = row  # Store the entire row as a dictionary value

    # Printing keys after populating the dictionary
    for key in data_dict.keys():
        print(key)

    return data_dict
'''

def load_ranks(file_path):
    entity_ranks = {}  # Initialize an empty dictionary to store the data

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Extracting necessary information from the row
            entity_id = row['Entity']
            entity_type = row['Entity_Type']
            ranks = row['Ranks'].split(', ') if 'Ranks' in row else []  # Assuming Ranks is comma-separated
            # Replace 0 with 1000 in the ranks list
            ranks = ['1000' if rank == '0' else rank for rank in ranks]

            # Creating a dictionary for the entity with its details
            entity_details = {'Entity_Type': entity_type, 'Ranks': ranks}
            
            # Saving entity details under the 'Entity' key
            entity_ranks[entity_id] = entity_details

    # Printing each item in the dictionary
    for key, value in entity_ranks.items():
        print(f"Key: {key}, Value: {value}")

    return entity_ranks

####################################################################################
###                                                                              ###
###                             Evaluation Metrics                               ###
###                                                                              ###
####################################################################################

# MEAN RANK
def mean_rank(ranks):

    # se a entidade so tiver um resultado, entao esse resultado e a Mean Rank dessa entidade
    #if len(ranks) == 1:
    #    rounded_mr = ranks[0]
    #    print(f"The list only has one value.")
    
    # se a entidade so tiver um resultado e for igual a zero, nao ha problema em considera-lo
    # pois o seu valor em nada afeta os cálculos
    
    #else:
    mr = np.mean(ranks) # np.average(ranks)
    rounded_mr = round(mr, 3)

    print('Mean Rank:', rounded_mr)

    return rounded_mr


# RECIPROCAL RANK
# only care about where relevant results ended up in the listing
def reciprocal_rank(ranks): # porque eu vou receber os ranks de cada entidade, em vez de um por um

    # Initiate empty list to store the reciprocal ranks
    reciprocal_ranks = []

    # Loop through the ranks
    for rank in ranks:
        # Check if rank is equal to 0
        if rank == 0:
            reciprocal_rank = 0 # ou rank
        else:
        # compute the reciprocal rank for each gene (NOW WE ONLY HAVE ONE)
            reciprocal_rank = 1 / (rank)

    rounded_rr = round(reciprocal_rank, 3)
    reciprocal_ranks.append(rounded_rr)

    return reciprocal_ranks


# MEAN RECIPROCAL RANK
def mean_reciprocal_rank(reciprocal_ranks):

    mrr = np.mean(reciprocal_ranks)
    rounded_mrr = round(mrr, 3)
    
    print('Mean Reciprocal Rank:', rounded_mrr)

    return rounded_mrr


# HITS AT K
def hits_at(ranks, k): # k:int
    hits = 0
    for rank in ranks:
        #print('rank', rank)
        if rank <= int(k):
            if rank != 0: # or hits != 0:
                hits += 1
    
    #print('k', k)
    print('HITS', hits)
    final_hits = float(hits) / float(len(ranks)) # proportion of HITS
    rounded_hits = round(final_hits, 3)
    print('HITS@', k, ':', rounded_hits)
    
    return rounded_hits

def new_hits(ranks, k):
    top = len([rank for rank in ranks if rank != 1000 and rank <= k])
    non_zeros = len([rank for rank in ranks if rank != 1000])
    print('HITS denominator:', non_zeros)
    final_hits = top/non_zeros
    rounded_hits = round(final_hits, 3)
    print('HITS@', k, ':', rounded_hits)

    return rounded_hits

####################################################################################
###                                                                              ###
###                             Mean for all entities                            ###
###                                                                              ###
####################################################################################

'''
def mean_entities(list_results):

    # receives a list and calculates the average of the values that are there
    mean_result = np.mean(list_results)
    rounded_result = round(mean_result, 3)
    
    return rounded_result
'''

####################################################################################
###                                                                              ###
###                             Put all to run                                   ###
###                                                                              ###
####################################################################################

def run_all(hs_value1, hs_value2, hs_value3, rank_files):

    # create a DataFrame to store all model names and metric results
    results = pd.DataFrame(columns=['Model', 'Mean Rank', 'Mean Reciprocal Rank', 
                                    'HITS@' + str(hs_value1), 'HITS@' + str(hs_value2), 'HITS@' + str(hs_value3)])
    
    for file in rank_files:

        # initiate empty list to store the intermediate values
        # goal: calculate the mean for an entity and all entities
        #mr_entities = []
        #mrr_entities = []
        #hs1_entities = []
        #hs2_entities = []
        #hs3_entities = []

        # get ranks
        entity_ranks = load_ranks(file)
        #entity_ranks = read_csv_to_dict(file)

        # Get the number of keys in the dictionary
        number_of_keys = len(entity_ranks)
        print("Number of keys in the dictionary:", number_of_keys)

        # Counting entities with at least one non-zero value
        num_entities_with_non_zero = sum(1 for entity_data in entity_ranks.values() if any(int(rank) != 0 for rank in entity_data['Ranks']))
        print(f"Number of entities with at least one non-zero result: {num_entities_with_non_zero}")

        # get model name
        model = file.split('_')[0]

        # get a list with all ranks from all entities
        #ranks_list = [entity[1] for entity in entity_ranks.values()]
            #ranks_list = [ranks for entity, (type_entity, ranks) in entity_ranks.items()]
        #ranks_list = [rank for entity, (type_entity, ranks) in entity_ranks.items() for rank in ranks]
        #print(ranks_list)
        # Extracting ranks and converting to integers
        all_ranks = [int(rank) for entity_data in entity_ranks.values() for rank in entity_data['Ranks'] if rank]
        print(all_ranks)
        print('The total number of GD and DG pairs is:', len(all_ranks))

        # Counting zeros and non-zeros using list comprehensions
        num_zeros = sum(1 for element in all_ranks if element == 1000)
        num_non_zeros = sum(1 for element in all_ranks if element != 1000)

        print(f"Number of unranked items: {num_zeros}")
        print(f"Number of found pairs: {num_non_zeros}")

        # calculate the performance (mean rank, mean reciprocal rank and HITS's) for each entity
        #for entity, (type_entity, ranks) in entity_ranks.items():
            # convert list of ranks to a NumPy array
        #convert_to_array = np.array(all_ranks)
            # print
            #print(convert_to_array.dtype)
            # fix the ranks data type
        #ranks_array = convert_to_array.astype(int)
            # print again
            #print(ranks_array.dtype)
            # calculate
        mr = mean_rank(all_ranks)
        rr = reciprocal_rank(all_ranks)
        mrr = mean_reciprocal_rank(rr)
        hs1 = new_hits(all_ranks, hs_value1) # proportion of
        hs2 = new_hits(all_ranks, hs_value2)
        hs3 = new_hits(all_ranks, hs_value3)
            # save in lists for all entities
            #mr_entities.append(mr_entity)
            #mrr_entities.append(mrr_entity)
            #hs1_entities.append(hs1_entity)
            #hs2_entities.append(hs2_entity)
            #hs3_entities.append(hs3_antity)
        
        #mr = mean_entities(mr_entities)
        #mrr = mean_entities(mrr_entities)
        #hs1 = mean_entities(hs1_entities)
        #hs2 = mean_entities(hs2_entities)
        #hs3 = mean_entities(hs3_entities)

        # add a new line to the results DataFrame with model name and performance
        row = {'Model': model, 'Mean Rank': mr, 'Mean Reciprocal Rank': mrr, 'HITS@' + str(hs_value1): hs1, 
               'HITS@' + str(hs_value2): hs2, 'HITS@' + str(hs_value3): hs3}
        results = results.append(row, ignore_index = True)
    
     # save the DataFrame to a CSV file
    results.to_csv('HITS_Evaluation_Results_1000.csv')
    
    print('----------------- End ------------------')

####################################################################################
###                                                                              ###
###                                    Main                                      ###
###                                                                              ###
####################################################################################

def main():
    parser = argparse.ArgumentParser(description="Script with three HITS@")
    parser.add_argument("--hs_value1", type=int, required=True, help="Specify K of the first HITS@")
    parser.add_argument("--hs_value2", type=int, required=True, help="Specify K of the second HITS@")
    parser.add_argument("--hs_value3", type=int, required=True, help="Specify K of the third HITS@")

    args = parser.parse_args()
    
    run_all(args.hs_value1, args.hs_value2, args.hs_value3, rank_files)

####################################################################################
###                                                                              ###
###                               Calling Functions                              ###
###                                                                              ###
####################################################################################

# Example of running
rank_files = ['TransE_Ranks.csv', 'TransD_Ranks.csv', 'TransH_Ranks.csv', 'TransR_Ranks.csv', 'distMult_Ranks.csv', 'HOLE_Ranks.csv', 'ComplEx_Ranks.csv']

# Call main function
if __name__ == "__main__":
    main()