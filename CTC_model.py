"""
#FOR GPU
from tensorflow.keras import backend as K_backend
if K_backend.backend() == 'tensorflow':
		import tensorflow as tf # Memory control with Tensorflow
		session_conf = tf.ConfigProto()
		#session_conf = tf.compat.v1.ConfigProto()
		session_conf.gpu_options.allow_growth=True
		sess = tf.Session(config=session_conf, graph=tf.get_default_graph())
		#sess = tf.compat.v1.Session(config=session_conf, graph=tf.compat.v1.get_default_graph())
		K_backend.set_session(sess)
		#tf.compat.v1.keras.backend.set_session(sess)
"""


import Code2Kern
import numpy as np
import editdistance
from itertools import groupby
import tensorflow.keras as K 
from tensorflow.keras import layers
from tensorflow.keras import activations

# from Decoders import *



"""## CTC Loss function"""
def ctc_lambda_func(args):
	y_pred, y_true, input_length, label_length = args
	return K.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


"""## Define the general CRNN model (training)"""
def create_train_model(symbol_set, yml_parameters):

	architecture = yml_parameters['architecture']

	#INPUT STAGE:
	in_img = layers.Input(
			shape = (yml_parameters['user_max_height'], None, 1),
			name = 'image',
			dtype = 'float32'
	)

	#CONVOLUTIONAL STAGE:
	x = in_img
	for conv_index in range(len(architecture['filters'])):
		x = layers.Conv2D(
			filters = architecture['filters'][conv_index],
			kernel_size = architecture['kernel_size'][conv_index],
			# strides = architecture['pool_strides'][conv_index],
			padding = 'same',
			name = 'Conv' + str(conv_index + 1) 
		)(x)
		if architecture['batch_norm'][conv_index]:
			x = layers.BatchNormalization(
				name = 'BatchNorm' + str(conv_index + 1) 
			)(x)
		x = eval('layers.' + architecture['activations'][conv_index] + '(' + \
				'name = ' + "'Activ" + str(conv_index + 1)  + "'" + \
				', alpha=' + str(architecture['param_activation'][conv_index]) + \
				')' + '(x)')
		# x = eval('activations.' + architecture['activations'][conv_index] + '(x)')
		x = layers.MaxPool2D(
				pool_size = architecture['pool_size'][conv_index],
				strides = architecture['pool_strides'][conv_index],
				padding = 'same',
				name = 'MaxPool' + str(conv_index + 1) 
		)(x)

	#AXES PERMUTATION FOR THE RNN NETWORK:
	x = layers.Permute((2,1,3))(x)
	
	#RESHAPING FOR THE RECURRENT STAGE:
	x_shape = x.shape
	x = layers.Reshape(
			target_shape=(-1, (x_shape[3]*x_shape[2])),
			name='Reshape'
	)(x)


	#RECURRENT STAGE:
	for rec_index in range(len(architecture['units'])):
		x = layers.Bidirectional(
				layers.LSTM(
						units = architecture['units'][rec_index],
						return_sequences = True,
						dropout = architecture['dropout'][rec_index],
						name='LSTM' + str(rec_index + 1) 
				),
				name='Bidirectional' + str(rec_index + 1) 
		)(x)
		if architecture['batch_norm_rec'][rec_index]:
			x = layers.BatchNormalization(
				name = 'BatchNormRec' + str(rec_index + 1) 
			)(x)

	#FINAL DENSE NN CLASSIFIER:
	y_pred = layers.Dense(
			units = len(symbol_set) + 1,
			activation = 'softmax',
			name = 'DenseClassifier'
	)(x)

	#AUXILIAR INPUTS FOR THE CTC MODEL:
	y_true = layers.Input(name='y_true', shape=[None], dtype='int64') 
	input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
	label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

	#INSTANTIATING THE LOSS:
	loss_out = layers.Lambda(
			function = ctc_lambda_func,
			output_shape = (1,),
			name = 'ctc'
	)([y_pred, y_true, input_length, label_length])

	#CREATING THE MODEL:
	model = K.Model(
			inputs=[in_img, y_true, input_length, label_length],
			outputs=[loss_out]
	)

	return model



"""##Define train and prediction models"""
def create_models(symbol_dict, yml_parameters, model_path):

	#Obtaining set of symbols from the dictionary:
	symbol_set = sorted(list(symbol_dict.keys()))
	
	#Train models:
	model = create_train_model(symbol_set, yml_parameters)

	#If path specified, loading weights from disk file:
	if model_path is not None:
		model.load_weights(model_path)

	model.compile(
		loss={'ctc': lambda y_true, y_pred: y_pred},
		optimizer = eval(yml_parameters['optimizer'])
	)
	model.summary()

	#Prediction model (same as train without CTC loss):
	prediction_model = K.Model(model.get_layer(name="image").input, model.get_layer(name='DenseClassifier').output)
	prediction_model.summary()

	return model, prediction_model


"""CTC prediction decoding function:"""
def ctc_manual_decoding(input_tensor, original_batch_images_size, yml_parameters):
	out = list()

	for it_batch in range(len(input_tensor)):
		#Performing Best Path decoding:
		decoded_sequence = list()
		probability_matrix = input_tensor[it_batch][0:original_batch_images_size[it_batch]]
		#Looping over temporal slices to analyze:
		for array in probability_matrix:
			#Estimated symbol:
			decoded_value = [np.where(array.max() == array)[0][0] if np.where(array.max() == array)[0][0] != len(array) -1 else -1]

			#Appending symbol:
			decoded_sequence.extend(decoded_value)
		
		#Applying function B for grouping alike symbols:
		decoded_sequence = [i[0] for i in groupby(decoded_sequence) if i[0] != -1]

		decoded_sequence.extend([-1]*(input_tensor.shape[1] - len(decoded_sequence)))
		out.append(np.array(decoded_sequence))
	return np.array(out)


"""Error functions"""
def error_functions(result_CTC_Decoding, y_true, y_true_symbol_length, inverse_symbol_dict, files = []):
	# Obtaining results:
	SeqER = 0
	SymER = 0
	SeqER_kern = 0
	SymER_kern = 0
	for it_seq in range(len(y_true)):
		if len(files) > 0:
			print("**** Assessing file {}".format(files[it_seq]))
		# Decoding:	
		CTC_prediction = [inverse_symbol_dict[str(u)] for u in np.array(result_CTC_Decoding[it_seq]) if u != -1]
		true_labels = [inverse_symbol_dict[str(u)] for u in y_true[it_seq][:y_true_symbol_length[it_seq]]]

		print("CTC prediction Before Code2Kern:\n -->{}".format(CTC_prediction))
		print("CTC prediction After Code2Kern:\n -->{}".format(Code2Kern.decode_prediction(CTC_prediction)))

		print("GT Before Code2Kern:\n -->{}".format(true_labels))
		print("GT After Code2Kern:\n -->{}".format(Code2Kern.decode_prediction(true_labels)))

		CTC_prediction_kern = Code2Kern.decode_prediction(CTC_prediction)
		true_labels_kern = Code2Kern.decode_prediction(true_labels)

		# Sequence error rate:
		if CTC_prediction != true_labels:
			SeqER = SeqER + 1

		# Sequence error rate in Kern:
		if CTC_prediction_kern != true_labels_kern:
			SeqER_kern = SeqER_kern + 1

		# Symbol error rate:
		SymER = SymER + editdistance.distance(true_labels,CTC_prediction)/float(len(true_labels))

		# Symbol error rate in Kern:
		SymER_kern = SymER_kern + editdistance.distance(true_labels_kern,CTC_prediction_kern)/float(len(true_labels_kern))

	# Normlization:
	SeqER = 100*SeqER/len(y_true)
	SymER = 100*SymER/len(y_true)

	# Normalization in kern:
	SeqER_kern = 100*SeqER_kern/len(y_true)
	SymER_kern = 100*SymER_kern/len(y_true)

	return SeqER, SymER, SeqER_kern, SymER_kern


def error_functions_batch(result_CTC_Decoding, y_true, y_true_symbol_length, inverse_symbol_dict, files = []):
    	# Obtaining results:
	SeqER = list()
	SymER = list()

	for it_seq in range(len(y_true)):
		# Decoding:	
		CTC_prediction = [inverse_symbol_dict[str(u)] for u in np.array(result_CTC_Decoding[it_seq]) if u != -1]
		true_labels = [inverse_symbol_dict[str(u)] for u in y_true[it_seq][:y_true_symbol_length[it_seq]]]

		# Sequence error rate:
		if CTC_prediction != true_labels:
			SeqER.append(1)
		else:
			SeqER.append(0)

		# Symbol error rate:
		SymER.append(editdistance.distance(true_labels,CTC_prediction)/float(len(true_labels)))

	return SeqER, SymER



def error_functions_batch_with_Kern_Reconstruction(result_CTC_Decoding, y_true, y_true_symbol_length, inverse_symbol_dict, files = []):
    	# Obtaining results:
	SeqER = list()
	SymER = list()
	SeqER_kern = list()
	SymER_kern = list()
	for it_seq in range(len(y_true)):
		# if len(files) > 0:
		# 	print("**** Assessing file {}".format(files[it_seq]))
		# Decoding:	
		CTC_prediction = [inverse_symbol_dict[str(u)] for u in np.array(result_CTC_Decoding[it_seq]) if u != -1]
		true_labels = [inverse_symbol_dict[str(u)] for u in y_true[it_seq][:y_true_symbol_length[it_seq]]]


		CTC_prediction_kern = Code2Kern.decode_prediction(CTC_prediction)
		true_labels_kern = Code2Kern.decode_prediction(true_labels)

		# Sequence error rate:
		if CTC_prediction != true_labels:
			SeqER.append(1)
		else:
			SeqER.append(0)

		# Sequence error rate in Kern:
		if CTC_prediction_kern != true_labels_kern:
			SeqER_kern.append(1)
		else:
			SeqER_kern.append(0)

		# Symbol error rate:
		SymER.append(editdistance.distance(true_labels,CTC_prediction)/float(len(true_labels)))

		# Symbol error rate in Kern:
		SymER_kern.append(editdistance.distance(true_labels_kern,CTC_prediction_kern)/float(len(true_labels_kern)))

	return SeqER, SymER, SeqER_kern, SymER_kern




"""Error functions (manual checking)"""
def error_functions_manual_checking(CTC_prediction, true_labels):
	# Obtaining results:
	SeqER = 0
	SymER = 0
	# Sequence error rate:
	if CTC_prediction != true_labels:
		SeqER = 100
	
	# Symbol error rate:
	SymER = 100 * editdistance.distance(true_labels,CTC_prediction)/float(len(true_labels))
	
	return SeqER, SymER


"""Load .h5 model"""
def load_disk_model(model_path):
	return K.models.load_model(model_path)