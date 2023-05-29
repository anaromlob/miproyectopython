"""
#FOR GPU
import sys, os, warnings
gpu = sys.argv[ sys.argv.index('-gpu') + 1 ] if '-gpu' in sys.argv else '0'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow CUDA load statements
warnings.filterwarnings('ignore')
"""

import numpy as np
import os
import shutil
import subprocess
import yaml
from flask import Blueprint, render_template, request, redirect, url_for
from music21 import converter

import CTC_model
import Data_processes
import test
import train

os.environ['musicxmlPath'] = '/Applications/Muse Hub.app/Contents/MacOS/Muse Hub'

INPUT_MODES = ['train', 'test', 'eval_set', 'export_set']
views = Blueprint(__name__, "views")

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

def mainmenu(filename):
    #config = menu()
    configconf = 'Primus/Primus.yml'
    configfold = 'Primus/Folds/Fold0'
    configmode = 'test'
    config_model_path = 'Primus/Folds/Fold0/Model/Primus.h5'
    input_path = 'Primus/Data/Audios/' + filename

    #Process configuration file:
    yml_parameters = process_configuration_file(configconf, configfold)

    if configmode == 'train':
        #Obtaining symbol dictionaries:
        #print("Obtaining symbol data...")
        if yml_parameters['create_dictionaries']:
            symbol_dict, inverse_symbol_dict = Data_processes.create_symbol_data(yml_parameters)
        else:
            symbol_dict, inverse_symbol_dict = Data_processes.retrieve_symbols(yml_parameters)

        #Creation of the models:
        model, prediction_model = CTC_model.create_models(symbol_dict, yml_parameters, config_model_path)

        #Training model:
        train.train_model(yml_parameters, model, prediction_model, symbol_dict, inverse_symbol_dict)

    elif configmode == 'test':
        prediccion = test.test_model(config_model_path, input_path, yml_parameters)

    elif configmode == 'eval_set':
        test.test_model_with_entire_set(config_model_path, input_path, yml_parameters)

    elif configmode == 'export_set':
        test.predict_export_entire_partition(config_model_path, input_path, yml_parameters)

    return prediccion


@views.route("/")
def home():
    return render_template("programa.html")

@views.route("/profile")
def profile():
    return render_template("profile.html")

@views.route('/predict/<string:audio>',methods=['GET'])
def predict(audio):
    prediction = mainmenu(audio)
    return redirect(url_for('views.partitura', mensaje=prediction))

@views.route('/partitura')
def partitura():
    mensaje = request.args.get('mensaje')
    newmensaje = mensaje.replace("Pred:", "")
    newnewmensaje = '**kern\n' + newmensaje.replace(",","\n").replace("	","")
   # archivo = open('prediccion.txt', 'w')
    #archivo.write(newnewmensaje)
    #archivo.close()

    score = converter.parse(newnewmensaje, format='humdrum')
    score.write('musicxml', 'partitura.xml')

    score_new = converter.parse('partitura.xml')

    subprocess.call(['/Applications/MuseScore 4.app/Contents/MacOS/mscore', '-o', 'partitura.png', 'partitura.xml'])
    ruta_origen = 'partitura-1.png'
    ruta_destino = "static/" + ruta_origen
    shutil.copy(ruta_origen, ruta_destino)

    return redirect(url_for('views.home'))

@views.route('/infoTec')
def infoTec():
    return render_template("infoTec.html")

@views.route('/infoData')
def infoData():
    return render_template("infoData.html")

@views.route('/index')
def index():
    return render_template("inicio.html")


def convertir(mensaje):
    mensaje = mensaje.replace('Pred:', '')
    simbolos = mensaje.split(',')
    return simbolos[0]