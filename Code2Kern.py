
"""Convert CRNN prediction into Kern"""
def decode_prediction(input_seq):
	out_seq = list()

	it = 0

	while it < len(input_seq):
		# print(input_seq[it])


		if "clef" in input_seq[it]: #CLEFs:
			out_seq.append(input_seq[it])
			it += 1

		elif "k[" in input_seq[it]: #KEYs:
			out_seq.append(input_seq[it])
			it += 1

		elif "*M" in input_seq[it]: #METERs:
			out_seq.append(input_seq[it])
			it += 1

		elif "=" in input_seq[it]: #SILENCEs:
			out_seq.append(input_seq[it])
			it += 1

		elif "s" in input_seq[it]: #SLURs:
			out_seq.append(input_seq[it])
			it += 1

		elif 'P' in input_seq[it]: #NOTEs:
			pitch = ''
			pitch_alteration = ''
			octave = ''
			duration_alteration = ''
			if len(input_seq[it]) == 2 and input_seq[it][0] == 'P' and input_seq[it][1] != 'A':
				# Note case:
				out_note = list()
				
				# Pitch:
				pitch = input_seq[it][1]
				it += 1

				# Alteration?
				pitch_alteration = ''
				if it < len(input_seq) and 'PA' in input_seq[it]:
					pitch_alteration = input_seq[it][2]
					out_note.append(pitch_alteration)
					it += 1
				
				# Octave?
				octave = ''
				if it < len(input_seq) and 'O' in input_seq[it]:
					octave = input_seq[it][1:]
					out_note.append(octave)
					it += 1

				# Duration?
				duration = ''
				if it < len(input_seq) and 'D' in input_seq[it]:
					duration = input_seq[it][1:]
					out_note.append(duration)
					it += 1

				# Duration alteration?
				duration_alteration = ''
				if it < len(input_seq) and 'DA' in input_seq[it]:
					duration_alteration = input_seq[it][2]
					out_note.append(duration_alteration)
					it += 1
			
			#Creating the actual note with the elements extracted:
			if pitch != 'r' and pitch != '':# and ('PA' not in input_seq[it] and 'D' not in input_seq[it] and 'O' not in input_seq[it]):
				if duration != '' and octave != '': #Non-silence
					note = pitchOctave2Kern(pitch, 'O'+octave)
					duration = duration + duration_alteration if len(duration_alteration) > 0 else duration
					note = duration + note

					note = note + pitch_alteration if len(pitch_alteration) > 0 else note

					# print(f'{out_note} <-> {note}')

					out_seq.append(note)

			elif 'r' in pitch: #Silence
				if duration != '':
					silence = duration + pitch
					out_seq.append(silence)
			else:
				it += 1
		else:
			it += 1
			
	return out_seq


"""Octave codification to Kern format"""
def pitchOctave2Kern(in_pitch, in_octave):
	# Extracting octave:
	octave = int("".join(in_octave.split("O")[1:]))

	if octave < 4:
		return "".join([in_pitch.upper() for u in range(4-octave)])
	else:
		return "".join([in_pitch.lower() for u in range(octave - 3)])

	




if __name__ == '__main__':
	print("hello")
	GT = ['*clefG2','*k[b-e-a-]','*M3/4','Pb','PA-','O5','D4','Pe','PA-','O5','D8','Pb','PA-','O5','D8','Pc','O6','D8','Pb','PA-','O5','D8','=','Pa','PA-','O5','D8','Pa','PA-','O5','D8','Pr','D16','Pa','PA-','O5','D16','Pg','O5','D16','Pa','PA-','O5','D16','Pb','PA-','O5','D16','Pa','PA-','O5','D16','Pg','O5','D16','Pa','PA-','O5','D16','=']
	GT = ['*clefG2','*k[b-e-a-]','*M3/4','Pb','PA-','O5','D4','Pe','PA-','O5','D8','Pb','PA-','O5','D8','Pc','O6','D8','Pb','PA-','O5','D8','=','Pa','PA-','O5','D8','Pa','PA-','O5','D8','Pr','D16','Pa','PA-','O5','D16','Pg','O5','D16','Pa','PA-','O5','D16','Pb','PA-','O5','D16','Pa','PA-','O5','D16','Pg','O5','D16','Pa','PA-','O5','D16','=']
	Pred = ['*clefC1','*k[b-e-a-]','Pr','D4','Pr','D4','Pr','D4','Pg','D8','Pg','D8','Pg','D16','D8','Pg','Pr','=']
	
	GT = ['*clefG2','*k[b-e-a-]','*M3/4','Pb','PA-','O5','D4','Pe','PA-','O5','D8','Pb','PA-','O5','D8','Pc','O6','D8','Pb','PA-','O5','D8','=','Pa','PA-','O5','D8','Pa','PA-','O5','D8','Pr','D16','Pa','PA-','O5','D16','Pg','O5','D16','Pa','PA-','O5','D16','Pb','PA-','O5','D16','Pa','PA-','O5','D16','Pg','O5','D16','Pa','PA-','O5','D16','=']
	Pred = ['*clefG2','*k[f#c#g#d#]','*M3/8','Pb','O5','D4','Pe','PA-','O5','D8','Pb','O5','D8','Pc','O6','Pb','O5','Pa','O5','D8','Pa','O5','Pa','D16','Pg','O5','D16']

	Pred = ['*clefC1','Pr','D4','=','Pr','D4','Pr','D8','Pg','O4','D8','D16','Pg','O4','D16','Pg','O4','D16','Pg','O4','D8','Pa','O4','D8','=','Pb','O4','D8','Pb','O4','D8','Pb','PA-','O4','D8','Pb','PA-','O4','D8','Pb','O4','D8','Pa','O4','D8','Pb','O4','D8','=','Pg','O4','Pg','O4','D4','=']
	GT = ['*clefC1','*k[b-e-a-]','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D2','Pg','O4','D8','Pg','O4','D16','Pg','O4','D16','Pg','O4','D8','Pa','PAn','O4','D8','=','Pb','PAn','O4','D8','Pb','O4','D8','Pr','D8','Pb','O4','D8','Pb','O4','D8','Pb','O4','D8','Pa','PAn','O4','D8','Pb','O4','D8','=','Pg','O4','D4','=']

	Pred = ['clefC1','*k[f#c#]','Pr','D4','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D2','Pr','D4','Pd','O4','D4','=','Pd','O4','D2','DA.','Pd','O4','D4','Pa','O4','D4','Pa','O4','D4','Pr','D4','Pa','O4','D8','Pa','O4','D8','=','Pf','PA#','O5','D2','Pd','O5','D4','Pd','O5','D4','=']
	GT = ['*clefC1','*k[b-]','Pr','D4','Pr','D2','Pr','D4','Pd','O4','D4','=','Pd','O4','D2','DA.','Pd','O4','D4','=','Pa','O4','D4','Pa','O4','D4','Pr','D4','Pa','O4','D8','Pa','O4','D8','=','Pf','O5','D2','Pd','O5','D4','Pd','O5','D4','=']

	Pred = ['*clefC1','Pr','D4','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D8','Pg','O4','D8','=','Pc','O5','D4','Pr','D4','Pr','D8','Pr','D8','Pc','O5','D8','Pe','O5','D4','Pc','O5','D8','=','Pa','O4','D4','Pa','O4','D8','=']
	GT = ['*clefC1','*M6/8','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','=','Pr','D4','Pr','D8','Pr','D4','Pg','O4','D8','=','Pc','O5','D4','Pr','D8','Pr','D4','Pr','D8','=','Pr','D4','Pc','O5','D8','Pe','O5','D4','Pc','O5','D8','=','Pa','O4','D4','Pa','O4','D8','Pr','D4','Pr','D8','=']

	GT = ['*clefC1','*k[b-e-a-]','*M3/4','Pr','D2','=','Pb','PA-','O4','D4','Pe','PA-','O4','D8','Pe','PA-','O5','D8','Pe','PA-','O5','D8','Pc','O5','D8','=','Pa','PA-','O4','D8','Pa','PA-','O4','D8','Pr','D4','Pr','D4','=','Pf','O5','D4','Pe','PA-','O5','D8','DA.','Pc','O5','D16','Pb','PA-','O4','D8','DA.','Pa','PA-','O4','D16','=','Pa','PA-','O4','D8','Pg','O4','D8','Pr','D4','=']

	Prediction = ['*clefG2', '*k[f#c#g#]', 'Pa', 'O4', 'D32', 'Pa', 'O4', 'D8', 'Pa', 'O4', 'D4', 'Pb', 'O4', 'D16', 'Pc', 'PA#', 'O5', 'D16', 'Pd', 'O5', 'D8', 'Pd', 'O5', 'D4', 'Pc', 'PA#', 'O5', 'D16', 'Pd', 'O5', 'D16', 'Pb', 'O4', 'D8', 'Pa', 'O4', 'D8', 'Pr', 'D32', 'Pe', 'O5', 'D32', '=', 'Pa', 'O5', 'D8', 'DA.', 'Pb', 'O5', 'D32', 'Pa', 'O5', 'D32', 'O5', 'D32', 'O5', 'D32', 'Pb', 'O5', 'D32', 'Pc', 'PA#', 'O6', 'D32', 'PA#', 'O6', 'D32', 'Pd', 'O6', 'D32', 'Pc', 'O6', 'D32', 'Pb', 'O5', 'Pa', 'O5', 'D8', 'Pr', 'D8', '=']

	decode_prediction(Prediction)
	# print(pitchOctave2Kern('g','O5'))
	
	# Pitch - p. alteration - Octave - Duration
	# print(pitchOctave2Kern('b','O5'))
	# ['b', '-', '5', '4']
	