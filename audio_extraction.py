import joblib
import essentia 
import numpy as np
import essentia.standard as es


#Joblib settings:
memory = joblib.memory.Memory('./joblib_cache', mmap_mode='r', verbose=1)

@memory.cache
def get_x_from_file(audiofilename):
	audio = es.MonoLoader(filename=audiofilename, sampleRate=44100)()
	windowing = es.Windowing(type='blackmanharris62', zeroPadding=4096)
	spectrum = es.Spectrum()
	spectrum_logfreq = es.LogSpectrum(frameSize=4096) # binsPerSemitone=1

	amp2db = es.UnaryOperator(type='lin2db', scale=2)
	pool = essentia.Pool()

	for frame in es.FrameGenerator(audio, frameSize=4096, hopSize=1024):
		frame_spectrum = spectrum(windowing(frame))
		frame_spectrum_logfreq, _, _ = spectrum_logfreq(frame_spectrum)

		pool.add('spectrum_db', amp2db(frame_spectrum))
		pool.add('spectrum_logfreq_db', amp2db(frame_spectrum_logfreq))

	return pool['spectrum_logfreq_db']


if __name__ == '__main__':
	x = get_x_from_file('descarga.mp3')
	x = np.flip(np.transpose(x),0)
	x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
	print("hello")