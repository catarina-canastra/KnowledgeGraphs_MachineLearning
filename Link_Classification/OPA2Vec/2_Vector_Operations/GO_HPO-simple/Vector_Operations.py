# -*- coding: utf-8 -*-
'''
    File name: Vector_operations.py
    Authors: Susana Nunes, Rita T. Sousa, Catia Pesquita
    Python Version: 3.7
'''
import numpy as np
import csv
import ast
from sklearn.metrics.pairwise import cosine_similarity


################################################
##           FILE MANIPULATION                ##
################################################

# CSV file reading function with pairs and ssm
def process_dataset_file(file_dataset_path):
    dataset = open(file_dataset_path, 'r')
    data = dataset.readlines()
    labels = []
    labels_list = []
    pairs = []
    url_pairs = []

    for line in data[1:]:
        nr, gene, disease, label = line.strip(";\n").split(";")
        labels.append(int(label))
        pairs.append((gene, disease))
        url_gene = "http://purl.obolibrary.org/obo/" + gene
        url_disease = "http://purl.obolibrary.org/obo/" + disease
        labels_list.append([(url_gene, url_disease), int(label)])
        url_pairs.append((url_gene, url_disease))

    dataset.close()
    return labels_list


# Function for reading the txt file with embeddings created with OPA2VEC
def process_opa2vec_embeddings_file(file_opa2vec_embeddings_path): # file_embeddings_path
    dataset = open(file_opa2vec_embeddings_path, 'r') # file_embeddings_path
    data = dataset.readlines()
    entities = []
    new_entities = [] # ADD
    vectors = []
    for line in data:
        values = line.strip('\n').split(' ')
        values = ' '.join(values).split()
        entities.append(values[0])
        #ent_0 = entities[0]
        #ent_0 = "http://purl.obolibrary.org/obo/" + ent_0
        #for ent in entities:
        #    ent = "http://purl.obolibrary.org/obo/" + ent
        url = "http://purl.obolibrary.org/obo/"
        new_entities = [url + ent for ent in entities]
        y = [float(i) for i in values[1:]]
        vectors.append(y)

    zip_iterator = zip(new_entities, vectors)
    dict_kg = dict(zip_iterator)

    dataset.close()
    return dict_kg


# Function for reading the txt file with embeddings created with RDF2Vec, OWL2Vec, DistMult, TransE
def process_other_embeddings_file(file_embeddings_path):
    dataset = open(file_embeddings_path, 'r')
    data = dataset.read()
    kgs = ast.literal_eval(data)

    dataset.close()
    return kgs


# Comparison of the pair file with the embeddings file, to associate each pair 2 embeddings (gene and disease).
def compare_files(labels_list, kgs):
    print("................. Comparing Files ................. ")
    pairs_kgs_label = []

    for ent in labels_list:

        if ent[0][0] in kgs and ent[0][1] in kgs:
            pairs_kgs_label.append([(ent[0][0], ent[0][1]), ([kgs[ent[0][0]]], [kgs[ent[0][1]]]), (ent[1])])

    print(' ................. Files compared ................. ')
    return pairs_kgs_label


############## NEW COMPARE FILES #################
def compare_files_opa2vec(labels_list, dict_kg):
    print("................. Comparing Files to OPA2Vec ................. ")
    pairs_kgs_label_opa2vec = []
    list_keys = list(dict_kg.keys())
    
    for ent in labels_list: # EDIT!!!!
    
        if ent[0][0] in list_keys and ent[0][1] in list_keys:
            pairs_kgs_label_opa2vec.append([(ent[0][0], ent[0][1]), ([dict_kg[str(ent[0][0])]], [dict_kg[str(ent[0][1])]]), (ent[1])])
    
    print(' ................. Files compared to OPA2Vec ................. ')
    return pairs_kgs_label_opa2vec



##############################################
##              OPERATIONS                  ##
##############################################

# Concatenation Operation - Now have an embedding with 400 features
def concatenation_operation(pairs):
    print('................. Concatenation operation ................. ')
    concat_input = []

    for ent in pairs:
        kg = ent[1][0] + ent[1][1]
        concat_input.append((ent[0], kg, ent[2]))

    delimiter_type = ';'
    with open('Data_Concatenated.csv', 'w', newline='') as concat_file:
        concat = csv.writer(concat_file, delimiter=delimiter_type)
        concat.writerow(["Entity 1", "Entity 2", "KG - Concatenation", "Label"])

        for ent in concat_input:
            concat.writerow([ent[0][0], ent[0][1], str(ent[1]).replace('[', '').replace(']', ''),
                             ent[2]])  # Then remove the [] directly in the csv (first place it in columns).

    concat_file.close()
    print('................. Concatenation finalized ................. ')
    return concat_input


# Average Operation- mean (1+1/2)
def average_operation(pairs):
    print('................. average operation ................. ')
    average_input = []

    for ent in pairs:
        kg1 = np.array(ent[1][0])
        kg2 = np.array(ent[1][1])
        kg = np.mean((kg1, kg2), axis=0)
        final_kg = np.ndarray.tolist(kg)
        average_input.append((ent[0], final_kg, ent[2]))

    delimiter_type = ';'
    with open('Data_Average.csv', 'w', newline='') as average_file:
        average = csv.writer(average_file, delimiter=delimiter_type)
        average.writerow(["Entity 1", "Entity 2", "KG - Average", "Label"])

        for ent in average_input:
            average.writerow([ent[0][0], ent[0][1], str(ent[1]).replace('[', '').replace(']', ''),
                              ent[2]])  # Then remove the [] directly in the csv (first place it in columns).
    len(average_input)
    average_file.close()
    print('................. average finalized ................. ')
    return average_input


# Hadamard Operation - element-wise matrix multiplication
def hadamard_operation(pairs):
    print('................. hadamard operation................. ')
    hadamard_input = []

    for ent in pairs:
        kg1 = np.array(ent[1][0])
        kg2 = np.array(ent[1][1])
        kg = np.multiply(kg1, kg2)
        final_kg = np.ndarray.tolist(kg)
        hadamard_input.append((ent[0], final_kg, ent[2]))

    delimiter_type = ';'
    with open('Data_Hadamard.csv', 'w', newline='') as hadamard_file:
        hadamard = csv.writer(hadamard_file, delimiter=delimiter_type)
        hadamard.writerow(["Entity 1", "Entity 2", "KG - Hadamard", "Label"])

        for ent in hadamard_input:
            hadamard.writerow([ent[0][0], ent[0][1], str(ent[1]).replace('[', '').replace(']', ''),
                               ent[2]])  # Then remove the [] directly in the csv (first place it in columns).

    hadamard_file.close()
    print('................. hadamard finalized................. ')
    return hadamard_input


# Weighted-L1 Operation - L1 loss function (normalize)
def weighted_L1_operation(pairs):
    print('L1 operation')
    L1_input = []

    for ent in pairs:
        kg1 = np.array(ent[1][0])
        kg2 = np.array(ent[1][1])
        kg = np.abs(kg1 - kg2)
        final_kg = np.ndarray.tolist(kg)
        L1_input.append((ent[0], final_kg, ent[2]))

    delimiter_type = ';'
    with open('Data_Weighted_L1.csv', 'w', newline='') as L1_file:
        L1 = csv.writer(L1_file, delimiter=delimiter_type)
        L1.writerow(["Entity 1", "Entity 2", "KG - Weighted Loss L1", "Label"])

        for ent in L1_input:
            L1.writerow([ent[0][0], ent[0][1], str(ent[1]).replace('[', '').replace(']', ''),
                         ent[2]])  # Then remove the [] directly in the csv (first place it in columns).

    L1_file.close()
    print('................. L1 finalized................. ')
    return L1_input


# Weighted-L2 Operation- L2 loss function (normalize and square)
def weighted_L2_operation(pairs):
    print('................. L1 operation................. ')
    L2_input = []
    for ent in pairs:
        kg1 = np.array(ent[1][0])
        kg2 = np.array(ent[1][1])
        kg = np.square(kg1 - kg2)
        final_kg = np.ndarray.tolist(kg)
        L2_input.append((ent[0], final_kg, ent[2]))

    delimiter_type = ';'
    with open('Data_Weighted_L2.csv', 'w', newline='') as L2_file:
        L2 = csv.writer(L2_file, delimiter=delimiter_type)
        L2.writerow(["Entity 1", "Entity 2", "KG - Weighted Loss L1", "Label"])

        for ent in L2_input:
            L2.writerow([ent[0][0], ent[0][1], str(ent[1]).replace('[', '').replace(']', ''),
                         ent[2]])  # Then remove the [] directly in the csv (first place it in columns).

    L2_file.close()
    print('................. L2 finalized................. ')
    return L2_input


# Calculate Cosine Similarity between the two embeddings
def cosine_similarity_operation(pairs):
    print('................. Cosine operation .................')
    cosine_input = []

    for ent in pairs:
        kg1 = ent[1][0]
        kg2 = ent[1][1]
        kg = cosine_similarity(kg1, kg2)
        final_kg = np.ndarray.tolist(kg)
        cosine_input.append((ent[0], final_kg, ent[2]))

    delimiter_type = ';'
    with open('Data_Cosine_Similarity.csv', 'w', newline='') as cosine_file:
        cosine = csv.writer(cosine_file, delimiter=delimiter_type)
        cosine.writerow(["Entity 1", "Entity 2", "KG - Cosine Similarity", "Label"])

        for ent in cosine_input:
            cosine.writerow([ent[0][0], ent[0][1], str(ent[1]).replace('[', '').replace(']', ''),
                             ent[2]])  # Then remove the [] directly in the csv (first place it in columns).
    cosine_file.close()
    print('................. Cosine finalized ................. ')
    return cosine_input


##############################################
##              Run Operations              ##
##############################################
def run_opa2vec_operations(file_dataset_path, file_opa2vec_embeddings_path): # file_embeddings_path
    labels_list = process_dataset_file(file_dataset_path)
    dict_kg = process_opa2vec_embeddings_file(file_opa2vec_embeddings_path) # file_embeddings_path
    pairs = compare_files_opa2vec(labels_list, dict_kg) # compare_files
    #print(pairs[0])
    
    #one_ent = labels_list[0][0]
    #print('We wnat this:', one_ent)
    
    #other_ent = labels_list[0][1]
    #print('And this:', other_ent)
    
    #one_entitie = new_entities[0]
    #print('An entity in Entities (keys dict):', one_entitie)
    
    #first_key = list(dict_kg.keys())[0]
    #print('First key:', first_key)
    
    #first_value = list(dict_kg.values())[0]
    #print('First value:', first_value)
    
    #get_vector = dict_kg.get(one_ent)
    #print(get_vector)

    concatenation_operation(pairs)
    average_operation(pairs)
    hadamard_operation(pairs)
    weighted_L1_operation(pairs)
    weighted_L2_operation(pairs)
    cosine_similarity_operation(pairs)
    

def run_other_operations(file_dataset_path, file_embeddings_path):
    labels_list = process_dataset_file(file_dataset_path)
    
    #one_ent = labels_list[0]
    #print('Nós queremos saber se isto está no dicionário:' + str(one_ent[0][0]))
    #print('E nós queremos saber se isto está também no dicionário:' + str(one_ent[0][1]))
    #print('Portanto, nós queremos saber se este gene e esta doença estão no dicionário.')
    
    kgs = process_other_embeddings_file(file_embeddings_path)
    pairs = compare_files(labels_list, kgs)
    
    #print(str(pairs[0]))

    concatenation_operation(pairs)
    average_operation(pairs)
    hadamard_operation(pairs)
    weighted_L1_operation(pairs)
    weighted_L2_operation(pairs)
    cosine_similarity_operation(pairs)


file_dataset_path = "Dataset_Pairs_Label.csv"
file_opa2vec_embeddings_path = "GO_HPOsimple_OPA2Vec_Embeddings.txt"
#file_embeddings_path = "Examples_RDF2Vec_Embeddings_GO_HPsimple.txt"
run_opa2vec_operations(file_dataset_path, file_opa2vec_embeddings_path)
#run_other_operations(file_dataset_path, file_embeddings_path)

#labels_list = process_dataset_file(file_dataset_path)
#dict_kg = process_opa2vec_embeddings_file(file_opa2vec_embeddings_path)

#one_ent = labels_list[0]
#gene = str("<" + one_ent[0][0] + ">")
#print(gene)

#for entities, vectors in dict_kg.items():
#   print(entities, vectors)

#keysList = list(dict_kg.keys())[40]
#print(str(keysList))