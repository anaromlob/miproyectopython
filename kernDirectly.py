import os
import re

init_krn = [
	'*clefC1',
	'*k[b-e-a-]',
	'*M3/4',
	'rr10',
	'=',
	'4b-y',
	'8e-y',
	'8ee-y',
	'8ee-y',
	'8cc',
	'=',
	'8a-y',
	'8a-y',
	'4r',
	'4r',
	'=',
	'4ff',
	'8.ee-y',
	'16cc',
	'8.b-y',
	'16a-y',
	'=',
	'8a-y',
	'8g',
	'4r',
	'=-'
]




def clean_sequence(init_krn):
	out_seq = list()

	for single_element in init_krn:
		if "clef" in single_element:
			out_seq.append(single_element)

		elif "k[" in single_element:
			out_seq.append(single_element)
		
		elif "*M" in single_element:
			out_seq.append(single_element)

		elif single_element.startswith('='):
			out_seq.append('=')
		
		elif single_element.startswith('s'):
			out_seq.append('s')

		elif not single_element.startswith('*') and not single_element.startswith('!'):
			if not 'q' in single_element:
				if 'rr' in single_element:
					out_seq.extend(re.findall('rr[0-9]+', single_element))
				elif 'r' in single_element:
					out_seq.append(single_element.split('r')[0]+'r')
				else:
					out_seq.extend(re.findall('\d+[.]*[a-gA-G]+[n#-]*', single_element))
	return out_seq


def obtain_symbol_dictionaries(yml_parameters, files):
    
	symbol_list = list()
	#Iterate through files:
	for single_file in files:
		with open(os.path.join(yml_parameters['path_to_GT'], single_file)) as f:
			init_krn = f.read().splitlines()
	
		clean_krn = clean_sequence(init_krn)
		symbol_list.extend(clean_krn)
		symbol_list = list(set(symbol_list))

	symbol_list = sorted(symbol_list)

	return symbol_list

if __name__ == '__main__':

	yml_parameters = {
		'path_to_GT' : 'Primus/Data/GT'}
	
	files = [u for u in os.listdir(yml_parameters['path_to_GT']) if u.endswith('.skm')]
	symbol_list = obtain_symbol_dictionaries(yml_parameters, files)

	# clean_sequence(init_krn)