# -*- coding: utf-8 -*-
'''
    File name: Run_RDF2Vec_Embeddings.py
    Authors: Susana Nunes, Rita T. Sousa, Catia Pesquita
    Python Version: 3.7
    Original RDF2Vec implementation in https://github.com/IBCNServices/pyRDF2Vec
    -sequences generated using the Weisfeiler-Lehman algorithm
    -walks depth 8
    -limited number of 500
    -corpora of sequences were used to build a Skip-Gram model with the default parameters.
'''

import numpy
import os
from operator import itemgetter
import rdflib
from rdflib.namespace import RDF, OWL, RDFS
import json
from pyrdf2vec.graphs import kg
from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec

from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import WeisfeilerLehmanWalker

#################################################
#####     Create KG with 3 ontologies       #####
#################################################
def construct_3kg(ontology_1_file_path, ontology_2_file_path, ontology_3_file_path, annotations_1_file_path,
                 annotations_2_file_path, annotations_3_file_path, entities_file):
    Kg = rdflib.Graph()
    Kg.parse(ontology_1_file_path, format='xml')
    Kg.parse(ontology_2_file_path, format='xml')
    Kg.parse(ontology_3_file_path, format='xml')

    ents = [line.strip() for line in open(entities_file).readlines()] # list of genes and diseases - entities for which I want embeddings 
    
    file_annot_do = open(annotations_3_file_path, 'r')
    for annot in file_annot_do:
        ent, do_term_list = annot[:-1].split('\t')
        
        url_ent = "http://purl.obolibrary.org/obo/" + ent  # url are the initials of do
        
        for url_do_term in do_term_list.split(';'):
            Kg.add((rdflib.term.URIRef(url_ent), rdflib.term.URIRef('http://purl.obolibrary.org/obo/hasAnnotation'),
                    rdflib.term.URIRef(url_do_term)))
    
    file_annot_hpo = open(annotations_1_file_path, 'r')
    for annot in file_annot_hpo:
        ent, hpo_term_list = annot[:-1].split('\t')

        url_ent = "http://purl.obolibrary.org/obo/" + ent  # url are the initials of hpo

        for url_hpo_term in hpo_term_list.split(';'):
            Kg.add((rdflib.term.URIRef(url_ent), rdflib.term.URIRef('http://purl.obolibrary.org/obo/hasAnnotation'),
                    rdflib.term.URIRef(url_hpo_term)))

    file_annot_go = open(annotations_2_file_path, 'r')
    for annot in file_annot_go:
        ent, go_term_list = annot[:-1].split('\t')

        url_ent = "http://purl.obolibrary.org/obo/" + ent  # url  are the initials of GO

        for url_go_term in go_term_list.split(';'):
            Kg.add((rdflib.term.URIRef(url_ent), rdflib.term.URIRef('http://purl.obolibrary.org/obo/hasAnnotation'),
                    rdflib.term.URIRef(url_go_term)))        

    print('..... KG created ..... ')
    file_annot_do.close()
    file_annot_hpo.close()
    file_annot_go.close()
    return Kg, ents

#################################################
#####     Create KG with 2 ontologies       #####
#################################################
def construct_2kg(ontology_1_file_path, ontology_2_file_path, annotations_1_file_path,
                 annotations_2_file_path, entities_file):
    Kg = rdflib.Graph()
    Kg.parse(ontology_1_file_path, format='xml')
    Kg.parse(ontology_2_file_path, format='xml')

    ents = [line.strip() for line in open(entities_file).readlines()] # list of genes and diseases - entities for which I want embeddings 
    
    file_annot_hpo = open(annotations_1_file_path, 'r')
    for annot in file_annot_hpo:
        ent, hpo_term_list = annot[:-1].split('\t')

        url_ent = "http://purl.obolibrary.org/obo/" + ent  # url are the initials of hpo

        for url_hpo_term in hpo_term_list.split(';'):
            Kg.add((rdflib.term.URIRef(url_ent), rdflib.term.URIRef('http://purl.obolibrary.org/obo/hasAnnotation'),
                    rdflib.term.URIRef(url_hpo_term)))

    file_annot_go = open(annotations_2_file_path, 'r')
    for annot in file_annot_go:
        ent, go_term_list = annot[:-1].split('\t')

        url_ent = "http://purl.obolibrary.org/obo/" + ent  # url  are the initials of GO

        for url_go_term in go_term_list.split(';'):
            Kg.add((rdflib.term.URIRef(url_ent), rdflib.term.URIRef('http://purl.obolibrary.org/obo/hasAnnotation'),
                    rdflib.term.URIRef(url_go_term)))        

    print('..... KG created ..... ')
    file_annot_hpo.close()
    file_annot_go.close()
    return Kg, ents

###############################################
#####     Create KG with 1 ontology       #####
###############################################
def construct_1kg(ontology_file_path, annotations_file_path, entities_file): #For one ontology only
    kg = rdflib.Graph()
    kg.parse(ontology_file_path, format='xml')

    ents = [line.strip() for line in open(entities_file).readlines()] # list of genes and diseases - entities for which I want embeddings 

    file_annot = open(annotations_file_path, 'r')

    for annot in file_annot:
        ent, hp_term_list = annot[:-1].split('\t')

        url_ent = "http://purl.obolibrary.org/obo/" + ent  # url são are the initials of HPO

        for url_hp_term in hp_term_list.split(';'):
            kg.add((rdflib.term.URIRef(url_ent), rdflib.term.URIRef('http://purl.obolibrary.org/obo/hasAnnotation'),
                    rdflib.term.URIRef(url_hp_term)))

    file_annot.close()
    
    print('..... KG created ..... ')
    return kg, ents


########################################
#####           RDF2Vec            #####
########################################
def calculate_embeddings(Kg, ents, path_output, size_value, type_word2vec, n_walks):
    graph = kg.rdflib_to_kg(Kg)

    if type_word2vec == 'CBOW':
        sg_value = 0
    if type_word2vec == 'skip-gram':
        sg_value = 1

    print('----------------------------------------------------------------------------------------')
    print('Vector size: ' + str(size_value))
    print('Type Word2vec: ' + type_word2vec)


    transformer = RDF2VecTransformer(Word2Vec(vector_size=size_value, sg=sg_value),
     walkers=[WeisfeilerLehmanWalker(8, n_walks, UniformSampler())])
    embeddings = transformer.fit_transform(graph, ents)

    with open(path_output + 'Embeddings_' + '_rdf2vec_' + str(type_word2vec) + '.txt',
              'w') as file:
        file.write("{")
        first = False
        for i in range(len(ents)):
            if first:
                file.write(", '%s':%s" % (str(ents[i]), str(embeddings[i].tolist())))
            else:
                file.write("'%s':%s" % (str(ents[i]), str(embeddings[i].tolist())))
                first = True
            file.flush()
        file.write("}")


##############################################
##              Run Embeddings              ##
##############################################

def run_embedddings_3kg(ontology_1_file_path, ontology_2_file_path, ontolgy_3_file_path, annotations_1_file_path,
                    annotations_2_file_path, annotations_3_file_path, vector_sizes, types_word2vec, n_walks, path_output, entities_file):
    Kg, ents = construct_3kg(ontology_1_file_path, ontology_2_file_path, ontology_3_file_path, annotations_1_file_path,
                            annotations_2_file_path, annotations_3_file_path, entities_file)
    #ensure_dir(path_output)
    calculate_embeddings(Kg, ents, path_output, vector_sizes, types_word2vec, n_walks)

def run_embedddings_2kg(ontology_1_file_path, ontology_2_file_path, annotations_1_file_path,
                    annotations_2_file_path, vector_sizes, types_word2vec, n_walks, path_output, entities_file):
    Kg, ents = construct_2kg(ontology_1_file_path, ontology_2_file_path, annotations_1_file_path,
                            annotations_2_file_path, entities_file)
    #ensure_dir(path_output)
    calculate_embeddings(Kg, ents, path_output, vector_sizes, types_word2vec, n_walks)


def run_embedddings_1kg(ontology_file_path, annotations_file_path,
                     vector_sizes, types_word2vec, n_walks, path_output, entities_file): #ONE ONTOLOGY
    Kg, ents = construct_1kg(ontology_file_path,annotations_file_path, entities_file)
    #ensure_dir(path_output)
    calculate_embeddings(Kg, ents, path_output, vector_sizes, types_word2vec, n_walks)

    
#Example of running
vector_sizes = 200
n_walks = 500
types_word2vec = "skip-gram"
path_output = "Run_"
entities_file = "RDF2Vec_entities.txt" #File with one entity per line 
ontology_1_file_path = "GO-full.owl"
ontology_2_file_path = "HPOsimple_08_logicD_AMLcompound.owl"
ontology_3_file_path = "HumanDO.owl"
annotations_1_file_path = "RDF2vec_OpenKE_GO_annotations.tsv"
annotations_2_file_path = "RDF2vec_OpenKE_HP_annotations.tsv"
annotations_3_file_path = "annotations_DO.tsv"
#run_embedddings_1kg(ontology_file_path, annotations_file_path, vector_sizes, types_word2vec, n_walks, path_output, entities_file)
run_embedddings_2kg(ontology_1_file_path, ontology_2_file_path, annotations_1_file_path, annotations_2_file_path, vector_sizes, types_word2vec, n_walks, path_output, entities_file)
#run_embedddings_3kg(ontology_1_file_path, ontology_2_file_path, ontology_3_file_path, annotations_1_file_path, annotations_2_file_path, annotations_3_file_path, vector_sizes, types_word2vec, n_walks, path_output, entities_file)