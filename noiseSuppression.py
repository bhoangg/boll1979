import sys
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

if(len(sys.argv)<2):
    print("Usage: \n python noiseSuppression.py file.wav")
    exit()

noisefile = sys.argv[1]

fs, data = wavfile.read(noisefile)
# Windows parametres
Nwin = 1024;Nhop=Nwin/2;nfft = Nwin
# Signal size
Length_data = len(data)
Trames = 1+round((Length_data-Nwin)/Nhop)

f, t, STFT_f = signal.stft(data, fs, nperseg=Nwin)
F=f
T=t

#plt.pcolormesh(T, F, np.abs(STFT_f), vmin=0, vmax=2*np.sqrt(2), shading='gouraud')
#plt.show()

#Step1: Half- Wave Rectification
f, t, STFT_band = signal.stft(data[:Nwin], fs, nperseg=Nwin)

u = np.mean(np.abs(STFT_band))
H= 1 - (u/np.abs(STFT_f))
Hr = (H+np.abs(H))/2
Xf=Hr*STFT_f

#Step2:  Residual Noise Reduction
w=np.angle(STFT_band)
Nr = STFT_band - u*np.exp(w*1j);
max_Nr = np.max(np.abs(Nr));
S_res=np.zeros(Xf.size);

for i in range(1,STFT_band.shape[0]-1):
    for j in range(2,Trames-1):
        if np.abs(Xf[i][j])<max_Nr:
            vec = [np.abs(Xf[i][j-1]), np.abs(Xf[i][j]), np.abs(Xf[i][j+1])]
            m = np.min(vec)
            Xf[i][j]= m

# plt.pcolormesh(T, F, np.abs(Xf), vmin=0, vmax=2*np.sqrt(2), shading='gouraud')
# plt.show()

t, Xs = signal.istft(Xf, fs, nperseg=Nwin)
wavfile.write("output.wav", fs, Xs.astype(np.int16))
print("Writing result to output.wav ...")
print("done.")

