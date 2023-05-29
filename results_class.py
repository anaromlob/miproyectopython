

class ResultsClass:
	def __init__(self):
		self.best_epoch = -1
		self.error_dict = {
			'Validation' : {
				'SymER' : float('inf'),
				'SeqER' : float('inf'),
				'SymER_kern' : float('inf'),
				'SeqER_kern' : float('inf')
			},
			'Test': {
				'SymER' : float('inf'),
				'SeqER' : float('inf'),
				'SymER_kern' : float('inf'),
				'SeqER_kern' : float('inf')
			},
		}
		return

	def __str__(self):
		out = "Neural Network format\n"
		out += "VALIDATION\t\t=> SymER: {:.2f} % - SeqER: {:.2f} % ".format(self.error_dict['Validation']['SymER'], self.error_dict['Validation']['SeqER'])
		out += "\nTEST\t\t\t=> SymER: {:.2f} % - SeqER: {:.2f} %".format(self.error_dict['Test']['SymER'], self.error_dict['Test']['SeqER'])

		out += "\n\nKern format\n"
		out += "VALIDATION\t\t=> SymER: {:.2f} % - SeqER: {:.2f} % ".format(self.error_dict['Validation']['SymER_kern'], self.error_dict['Validation']['SeqER_kern'])
		out += "\nTEST\t\t\t=> SymER: {:.2f} % - SeqER: {:.2f} %".format(self.error_dict['Test']['SymER_kern'], self.error_dict['Test']['SeqER_kern'])

		return out



"""Main"""
if __name__=='__main__':

	#Folder to process:
	results = ResultsClass()
	print(results)
