# -*- coding: utf-8 -*-
#
# Script: Full_Train_Model.py
# 
# Autor: Catarina Canastra and Catia Pesquita
#
# Use cases: train link prediction embedding models of OpenKE library
#            and creates a folder to save each model checkpoint


####################################################################################
###                                                                              ###
###                             Libraries and Packages                           ###
###                                                                              ###
####################################################################################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#from memory_profiler import profile
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import rdflib
from rdflib.namespace import RDF, OWL, RDFS
import sys
import re
from OpenKE import config
from OpenKE import models

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


####################################################################################
###                                                                              ###
###                             Build and train model                            ###
###                                                                              ###
####################################################################################

def construct_model(path_output, vector_size, model):
    con = config.Config()
    con.set_in_path(path_output)
    con.set_dimension(vector_size)

    print('--------------------------------------------------------------------------------------')
    print('MODEL: ' + model)

    # Models will be exported via tf.Saver() automatically
    con.set_export_files(path_output + model + "/model.vec.tf", 0)
    # Model parameters will be exported to json files automatically
    con.set_out_files(path_output + model + "/embeddings.vec.json")

    # Define the tasks we are going to perform
    con.set_test_link_prediction(True)

    if model == 'ComplEx':
        con.set_work_threads(8)
        con.set_train_times(100) # 1000
        con.set_nbatches(100)
        con.set_alpha(0.5)
        con.set_lmbda(0.05)
        con.set_bern(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.ComplEx)

    elif model == 'distMult':
        con.set_work_threads(8)
        con.set_train_times(100) # 1000
        con.set_nbatches(100)
        con.set_alpha(0.5)
        con.set_lmbda(0.05)
        con.set_bern(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.DistMult)

    elif model == 'HOLE':
        con.set_work_threads(8) #4
        con.set_train_times(100) #500
        con.set_nbatches(100)
        con.set_alpha(0.1)
        con.set_bern(0)
        con.set_margin(0.2)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.HolE)

    elif model == 'RESCAL':
        con.set_work_threads(8) #4
        con.set_train_times(100) # 500
        con.set_nbatches(100)
        con.set_alpha(0.1)
        con.set_bern(0)
        con.set_margin(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.RESCAL)

    elif model == 'TransD':
        con.set_work_threads(8)
        con.set_train_times(100) # 1000
        con.set_nbatches(100)
        con.set_alpha(1.0)
        con.set_margin(4.0)
        con.set_bern(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransD)

    elif model == 'TransE':
        con.set_work_threads(8)
        con.set_train_times(100) # susana?
        con.set_nbatches(100)
        con.set_alpha(0.001)
        con.set_margin(1.0)
        con.set_bern(0)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransE)

    elif model == 'TransH':
        con.set_work_threads(8)
        con.set_train_times(100) # 1000
        con.set_nbatches(100)
        con.set_alpha(0.001)
        con.set_margin(1.0)
        con.set_bern(0)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransH)

    elif model == 'TransR':
        con.set_work_threads(8)
        con.set_train_times(100) # 1000
        con.set_nbatches(100)
        con.set_alpha(1.0)
        con.set_lmbda(4.0)
        con.set_margin(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransR)
    
    print('------------- Built Model --------------')

    # Train the model.
    con.run()

    print('------------- Trained Model ------------')


####################################################################################
###                                                                              ###
###                             Put all to run                                   ###
###                                                                              ###
####################################################################################


def train(vector_size, model_names):

    for model in model_names:

        # define paths
        path_output =  './' # OpenKE/   OTHER  ./Benchmark_Simulation_Folder/GO_HPO/
        path_model_json = './' + model + "/results.vec.json"
        path_embeddings_output = "/output/Run_Results_" + model

        # build and train model
        construct_model(path_output, vector_size, model)
    
    print('End')


####################################################################################
###                                                                              ###
###                               Calling Functions                              ###
###                                                                              ###
####################################################################################  

# Example of running

vector_size = 200
model_names = ['TransE', 'TransD', 'TransH', 'TransR', 'distMult', 'RESCAL', 'HOLE', 'ComplEx']

train(vector_size, model_names)