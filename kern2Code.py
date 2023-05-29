import os
import re

init_krn = [
	'*clefG2',
	'*k[f#c#g#]',
	'*M3/4',
	'8aL',
	'16cc#y',
	'16bJ',
	'8f#L',
	'8g#y',
	'8f#y',
	'8e#J',
	'=',
	'8f#yL',
	'16a',
	'16g#yJ',
	'8f#yL',
	'8e',
	'8d',
	'8c#yJ',
	'=',
	'8dL',
	'16c#y',
	'16BJ',
	'4e',
	'4e',
	'=',
	'8AL',
	'8e',
	'8f#y',
	'8g#y',
	'8a',
	'8bJ',
	'='
]


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


"""Processing notes and rests"""
def process_kern_note_to_dict(note, meters, mode):

	duration = re.findall('[0-9]+', note)
	duration = duration[0] if len(duration) == 1 else '4'
	dot = '.' if '.' in note else ''

	if 'rr' in note:
		if mode == 'extract_symbols' : # Multirest
			return 'O', 'O', 'O', 'O', 'O'
		elif mode == 'encode_symbols' : # Multirest
			# current_meter = meters[-1] if len(meters) > 0 else '*M4/4' # Either using the existing meter or assuming a 4/4
			# # Extracting numer of bars affected by the multirest:
			# bars_multirest = int(note.split('rr')[1])
			# multirest_out = list()
			# for single_bar in range(bars_multirest):
			# 	for _ in range(int(current_meter.split('M')[1].split("/")[0])):
			# 		multirest_out.extend(['Pr', 'PA', 'O', 'D' + current_meter.split("/")[1], 'DA'])
			# return multirest_out 

			# ALTERNATIVE : NOT CONSIDERING MULTIREST:
			return 'O', 'O', 'O', 'O', 'O'

	elif 'r' not in note : # Regular note
		#Obtaining pitch:
		pitch_raw = re.findall('[a-gA-G]+', note)[0]
		pitch = "".join(set(pitch_raw.lower()))

		#Obtaining octave:
		octave = ''
		if pitch_raw.isupper():
			octave = str(4 - len(pitch_raw))
		else:
			octave = str(len(pitch_raw) + 3)

		#Obtaining alteration:
		alteration = "".join(re.findall('[n#-]+', note))

		return 'P'+pitch, 'PA' + str(alteration), 'O'+str(octave), 'D'+str(duration), 'DA'+dot
		# return 'P'+pitch, 'PA' + str(alteration), str(octave), str(duration), 'DA'+dot

	else: # Simple rest
		return 'Pr', 'PA', 'O', 'D'+str(duration), 'DA'+dot
		# return 'Pr', 'PA', 'O', str(duration), 'DA'+dot


"""Obtaining kern dictionary using the corpora provided"""
def obtain_symbol_dictionaries(yml_parameters, files):

	clefs = list()
	keys = list()
	meters = list()
	pitches = list()
	pitch_alterations = list()
	octaves = list()
	durations = list()
	duration_alterations = list()

	#Iterate through files:
	for single_file in files:
		with open(os.path.join(yml_parameters['path_to_GT'], single_file)) as f:
			init_krn = f.read().splitlines()

		#CLEFs:
		clefs.extend([s for s in init_krn if "clef" in s])

		#KEYs:
		keys.extend([s for s in init_krn if "k[" in s])

		#METERs:
		meters.extend([s for s in init_krn if "*M" in s])

		#Music elements
		music_elements = [music_element for music_element in init_krn if not music_element.startswith('*') and not music_element.startswith('!')]

		for element in music_elements:
			if not 'q' in element and '=' not in element and 's' not in element:
				sal = process_kern_note_to_dict(element, meters, 'extract_symbols')
				pitches.append(sal[0])
				pitch_alterations.append(sal[1])
				octaves.append(sal[2])
				durations.append(sal[3])
				duration_alterations.append(sal[4])

	#Last additional symbols:
	symbol_list = list()
	symbol_list.append('s')

	#Creating symbol list:
	if yml_parameters['elements_in_GT']['clefs']:
		symbol_list.extend(list(set(clefs)))
	
	if yml_parameters['elements_in_GT']['keys']:	
		symbol_list.extend(list(set(keys)))

	if yml_parameters['elements_in_GT']['meters']:	
		symbol_list.extend(list(set(meters)))

	if yml_parameters['elements_in_GT']['barlines']:	
		symbol_list.append('=')

	symbol_list.extend(list(set(pitches)))
	symbol_list.extend(list(set(pitch_alterations)))
	symbol_list.extend(list(set(octaves)))
	symbol_list.extend(list(set(durations)))
	symbol_list.extend(list(set(duration_alterations)))

	#Obtaining the unique symbols (as a set):
	symbol_list = sorted(list(set(symbol_list)))

	#Removing non-relevant symbols:
	try:
		symbol_list.remove('DA')
	except:
		pass
	try:
		symbol_list.remove('O')
	except:
		pass
	try:
		symbol_list.remove('PA')
	except:
		pass
	
	#Obtaining the unique symbols (as a set):
	symbol_list = sorted(list(set(symbol_list)))

	return symbol_list


"""Formatting kern sequence to its use with the dictionary"""
def krnInitProcessing(init_krn, yml_parameters):
	out_seq = list()

	meters_aux = list() # For the multirest case

	for single_element in init_krn:
		if "clef" in single_element:
			if yml_parameters['elements_in_GT']['clefs']: #CLEF?
				out_seq.append(single_element)

		elif "k[" in single_element:
			if yml_parameters['elements_in_GT']['keys']: #KEY?
				out_seq.append(single_element)
		
		elif "*M" in single_element:
			meters_aux.append(single_element)
			if yml_parameters['elements_in_GT']['meters']: #METER?
				out_seq.append(single_element)

		elif single_element.startswith('='):
			if yml_parameters['elements_in_GT']['barlines']: #MUSIC ELEMENT?
				out_seq.append('=')
		
		elif single_element.startswith('s'):
			out_seq.append('s')

		elif not single_element.startswith('*') and not single_element.startswith('!'):
			if not 'q' in single_element:
				out_seq.extend(process_kern_note_to_dict(single_element, meters_aux, 'encode_symbols'))

	out_seq = list(filter(('PA').__ne__, out_seq))
	out_seq = list(filter(('O').__ne__, out_seq))
	out_seq = list(filter(('DA').__ne__, out_seq))

	return out_seq




if __name__ == '__main__':
	elements_in_GT = {
		'clefs' : True,
		'keys' : True,
		'meters' : True,
		'barlines' : True
	}

	yml_parameters = {
		'path_to_GT' : 'Primus/Data/GT',
		'elements_in_GT' : elements_in_GT}

	files = [u for u in os.listdir(yml_parameters['path_to_GT']) if u.endswith('.krn')]
	symbol_list = obtain_symbol_dictionaries(yml_parameters, files)



	out_seq = krnInitProcessing(init_krn, yml_parameters)

