
"""
#FOR GPU
import sys, os, warnings
gpu = sys.argv[ sys.argv.index('-gpu') + 1 ] if '-gpu' in sys.argv else '0'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow CUDA load statements
warnings.filterwarnings('ignore')
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for


import os
import argparse
import numpy as np
import yaml
import train
import test
import sys

import Data_processes
import CTC_model

INPUT_MODES = ['train', 'test', 'eval_set', 'export_set']


"""Arguments menu"""
def menu():
	parser = argparse.ArgumentParser(description='CRNN-CTC implementation')

	parser.add_argument('-mode',		dest="mode",		required=True,			help='Train/test CRNN-CTC', choices = INPUT_MODES)
	parser.add_argument('-model',		dest="model_path",	required=False,			help='Path of the .h5 model to load')
	parser.add_argument('-input',		dest="input_path",	required=False,			help='Path of the element (sound/entire set) to test')
	parser.add_argument('-conf',		dest="conf",		required=True,			help='YAML configuration file')
	parser.add_argument('-fold',		dest="fold",		required=True,			help='Fold to process', type=str)

	args = parser.parse_args()

	return args


"""Processing YAML configuration file"""
def process_configuration_file(configuration_file, fold):

	with open(configuration_file) as f:
		yml_parameters = yaml.load(f, Loader=yaml.FullLoader)


	#Width reduction:
	yml_parameters["architecture"]["width_reduction"] = np.prod([u[1] for u in yml_parameters["architecture"]["pool_strides"]])
	yml_parameters["architecture"]["height_reduction"] = np.prod([u[0] for u in yml_parameters["architecture"]["pool_strides"]])


	#Obtain tuples:
	yml_parameters["architecture"]["kernel_size"] = [tuple(v) for v in yml_parameters["architecture"]["kernel_size"]]
	yml_parameters["architecture"]["pool_size"] = [tuple(v) for v in yml_parameters["architecture"]["pool_size"]]
	yml_parameters["architecture"]["pool_strides"] = [tuple(v) for v in yml_parameters["architecture"]["pool_strides"]]

	#Additional paths:
	###Fold:
	yml_parameters['path_to_fold'] = os.path.join(yml_parameters['path_to_corpus'], 'Folds', 'Fold' + fold)

	###Partitions:
	yml_parameters['path_to_partitions'] = os.path.join(yml_parameters['path_to_fold'], 'Partitions')

	###Images:
	yml_parameters['path_to_audios'] = os.path.join(yml_parameters['path_to_corpus'], 'Data', 'Audios')
	yml_parameters['path_to_GT'] = os.path.join(yml_parameters['path_to_corpus'], 'Data', 'GT')

	###Model & weights:
	yml_parameters['path_to_model'] = os.path.join(yml_parameters['path_to_fold'], 'Model')
	yml_parameters['path_to_weights'] = os.path.join(yml_parameters['path_to_fold'], 'Weights')

	return yml_parameters

@views.route("/main")
def mainmenu(): 
	config = menu()
	#Process configuration file:
	yml_parameters = process_configuration_file(config.conf, config.fold)
	
	if config.mode == 'train':  
		#Obtaining symbol dictionaries:
		print("Obtaining symbol data...")
		if yml_parameters['create_dictionaries']:
			symbol_dict, inverse_symbol_dict = Data_processes.create_symbol_data(yml_parameters)
		else:
			symbol_dict, inverse_symbol_dict = Data_processes.retrieve_symbols(yml_parameters)

		#Creation of the models:
		model, prediction_model = CTC_model.create_models(symbol_dict, yml_parameters, config.model_path)

		#Training model:
		train.train_model(yml_parameters, model, prediction_model, symbol_dict, inverse_symbol_dict)

	elif config.mode == 'test':
		prediccion = test.test_model(config.model_path, config.input_path, yml_parameters)

	elif config.mode == 'eval_set':
		test.test_model_with_entire_set(config.model_path, config.input_path, yml_parameters)
	
	elif config.mode == 'export_set':
		test.predict_export_entire_partition(config.model_path, config.input_path, yml_parameters)

	return prediccion

"""Main"""
if __name__ == "__main__":
	config = mainmenu()