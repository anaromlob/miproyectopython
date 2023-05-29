import os
import numpy as np
import CTC_model as CTC_model
import Data_processes as Data_processes
import Code2Kern





"""Function for explicitly testing the trained model"""
def test_model(model_path, image_path, yml_parameters):
	#Loading the .h5 model:
	prediction_model = CTC_model.load_disk_model(model_path)
	#Loading selected image:
	file_name = os.path.basename(image_path)
	#Obtaining additional parameters required:
	symbol_dict, inverse_symbol_dict = Data_processes.retrieve_symbols(yml_parameters)
	
	#Reading image...
	X_test, X_test_len, GT_label = Data_processes.load_image_for_manual_test(image_name = file_name,\
		symbol_dict = symbol_dict, yml_parameters = yml_parameters)


	#Additional vectors for training:
	X_test_len[0] = X_test_len[0]//yml_parameters['architecture']['width_reduction']

	# Test predictions:
	y_test_prediction = prediction_model.predict(
		x = X_test
	)

	#Decoding test predictions (current group):
	result_CTC_Decoding = CTC_model.ctc_manual_decoding(y_test_prediction, X_test_len, yml_parameters)
	CTC_prediction = [inverse_symbol_dict[str(u)] for u in np.array(result_CTC_Decoding[0]) if u != -1]

	#Obtaining metrics:
	SeqER, SymER = CTC_model.error_functions_manual_checking(CTC_prediction = CTC_prediction, true_labels = GT_label)

	#Printing results:
	print("Pred:\t" + ",".join(CTC_prediction))
	string = "Pred:\t" + ",".join(CTC_prediction)

	print("GT:\t" + ",".join(GT_label))

	print("- Symbol error: " + str(SymER) + "\n- Sequence error: " + str(SeqER))

	if yml_parameters['GT_type'] == 'krn':
		print("Pred:\t{}".format(",".join(Code2Kern.decode_prediction(CTC_prediction))))
		print("GT:\t{}".format(",".join(Code2Kern.decode_prediction(GT_label))))

		#Obtaining metrics:
		SeqER, SymER = CTC_model.error_functions_manual_checking(CTC_prediction = Code2Kern.decode_prediction(CTC_prediction),\
			true_labels = Code2Kern.decode_prediction(GT_label))

		print("- Symbol error: " + str(SymER) + "\n- Sequence error: " + str(SeqER))

	return string



"""Function for testing with an entire set (train/validation/test)"""
def test_model_with_entire_set(model_path, partition, yml_parameters):
	#Loading the .h5 model:
	prediction_model = CTC_model.load_disk_model(model_path)

	#Obtaining additional parameters required:
	symbol_dict, inverse_symbol_dict = Data_processes.retrieve_symbols(yml_parameters)

	#Listing files:
	train_files, val_files, test_files = Data_processes.list_files(yml_parameters)
	files = val_files if partition == 'validation' else test_files

	SeqER_error_list = list()
	SymER_error_list = list()
	SeqER_kern_error_list = list()
	SymER_kern_error_list  = list()

	init_index = 0
	while init_index < len(files):
		
		end_index = min(init_index + yml_parameters['batch_size'], len(files))

		#Loading data:
		X, Y, X_len, Y_len = Data_processes.load_selected_range(init_index = init_index, end_index = end_index,\
										files = files, symbol_dict = symbol_dict, yml_parameters = yml_parameters)

		#Additional vectors for training:
		input_length_train = np.zeros([X.shape[0],], dtype='int64')
		for i in range (X.shape[0]):
			input_length_train[i] = X_len[i]//yml_parameters['architecture']['width_reduction']

		# Predictions (current group):
		y_prediction = prediction_model.predict(
			x = X
		)

		#Decoding test predictions (current group):
		result_CTC_Decoding = CTC_model.ctc_manual_decoding(y_prediction, input_length_train, yml_parameters)

		#Figures of merit:
		if yml_parameters['GT_type'] == 'krn':
			SeqER_error, SymER_error, SeqER_kern_error, SymER_kern_error = CTC_model.error_functions_batch_with_Kern_Reconstruction(result_CTC_Decoding, Y, Y_len, inverse_symbol_dict, files[init_index:end_index])
		else:
			SeqER_error, SymER_error, = CTC_model.error_functions_batch(result_CTC_Decoding, Y, Y_len, inverse_symbol_dict, files[init_index:end_index])
			SeqER_kern_error = [0]*len(SeqER_error)
			SymER_kern_error = [0]*len(SeqER_error)
		
		SeqER_error_list.extend(SeqER_error)
		SymER_error_list.extend(SymER_error)
		SeqER_kern_error_list.extend(SeqER_kern_error)
		SymER_kern_error_list.extend(SymER_kern_error)


		#Updating index:
		init_index = end_index

	#Figures of merit:
	SeqER_error = 100*sum(SeqER_error_list)/len(files)
	SymER_error = 100*sum(SymER_error_list)/len(files)
	SeqER_kern_error = 100*sum(SeqER_kern_error_list)/len(files)
	SymER_kern_error = 100*sum(SymER_kern_error_list)/len(files)

	out = 	"Results\n"
	out += "NN\t\t=> SymER: {:.2f} % - SeqER: {:.2f} % ".format(SymER_error, SeqER_error)
	out += "\nKern\t\t=> SymER: {:.2f} % - SeqER: {:.2f} % ".format(SymER_kern_error, SeqER_kern_error)

	print(out)

	return



def predict_export_entire_partition(model_path, partition, yml_parameters):

	#Loading the .h5 model:
	prediction_model = CTC_model.load_disk_model(model_path)

	#Obtaining additional parameters required:
	symbol_dict, inverse_symbol_dict = Data_processes.retrieve_symbols(yml_parameters)

	#Listing files:
	train_files, val_files, test_files = Data_processes.list_files(yml_parameters)
	files = val_files if partition == 'validation' else test_files


	gt_out = open(os.path.join(yml_parameters['path_to_corpus'], 'GT-' + partition + '.txt'), 'w')
	pred_out = open(os.path.join(yml_parameters['path_to_corpus'], 'Pred-' + partition + '.txt'), 'w')

	for it_file in range(len(files)):
		print("{} out of {}".format(it_file+1, len(files)))
		
		#Loading data:
		X, Y, X_len, Y_len = Data_processes.load_selected_range(init_index = it_file, end_index = it_file + 1,\
										files = files, symbol_dict = symbol_dict, yml_parameters = yml_parameters)

		#Additional vectors for training:
		input_length_train = np.zeros([X.shape[0],], dtype='int64')
		for i in range (X.shape[0]):
			input_length_train[i] = X_len[i]//yml_parameters['architecture']['width_reduction']

		# Predictions (current group):
		y_prediction = prediction_model.predict(
			x = X
		)

		#Decoding test predictions (current group):
		result_CTC_Decoding = CTC_model.ctc_manual_decoding(y_prediction, input_length_train, yml_parameters)

		CTC_prediction_kern = [inverse_symbol_dict[str(u)] for u in np.array(result_CTC_Decoding[0]) if u != -1]
		true_labels_kern = [inverse_symbol_dict[str(u)] for u in Y[0][:Y_len[0]]]

		if yml_parameters['GT_type'] == 'krn':
			true_labels_kern = Code2Kern.decode_prediction(true_labels_kern)
			CTC_prediction_kern = Code2Kern.decode_prediction(CTC_prediction_kern)


		pred_out.write(",".join(CTC_prediction_kern) + '\n')
		gt_out.write(",".join(true_labels_kern) + '\n')

	gt_out.close()
	pred_out.close()

	return