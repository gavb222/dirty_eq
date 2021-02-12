import torch
import torchaudio
import math

n_fft = 2047
window_fn = torch.hann_window(n_fft)
hop_sz = int((n_fft+1)/8)
fs = 16000

def sigmoid(dist):
    #spread controls the size of the array
    linspace = torch.linspace(-3,3,dist)
    linspace = torch.div(1,1+torch.pow(math.e,(-1 * linspace)))
    #print(linspace)
    return linspace

def gaussian(dist):
    #spread controls the size of the array
    linspace = torch.linspace(-3,3,dist)
    # gaussian = e^((-x)^2/2) when standard dev is 1 and height is 1
    linspace = torch.exp(-1 * torch.div(torch.pow(linspace,2),2))
    return linspace

def make_wav(mag,phase):
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    spec = torch.stack([real,imag], dim = -1)
    wav = torch.istft(spec, n_fft = n_fft, hop_length = hop_sz, window = window_fn)
    return wav

def eq(filepath,low_beta,lowmid_beta,highmid_beta,high_beta):
    wavData,fs = torchaudio.load(filepath)
    #print(fs)
    wavData = torch.mean(wavData, dim=0).unsqueeze(0)

    complex_mix = torch.stft(wavData, n_fft = n_fft, hop_length = hop_sz, window = window_fn)
    complex_mix_pow = complex_mix.pow(2).sum(-1)
    complex_mix_mag = torch.sqrt(complex_mix_pow)
    complex_mix_phase = torch.tan(torch.div(complex_mix[:,:,:,1],complex_mix[:,:,:,0]))
    #print(complex_mix.size())

    low_curve = (sigmoid(int((n_fft+1)/8))*low_beta)+(1-low_beta)
    lowmid_curve = (gaussian(int((n_fft+1)/8))*lowmid_beta)+1
    highmid_curve = (gaussian(int((n_fft+1)/8))*highmid_beta)+1
    high_curve = (sigmoid(int((n_fft+1)/8))*high_beta*-1)+1

    overall_curve = torch.cat((low_curve,lowmid_curve,highmid_curve,high_curve),dim=0)

    x,y = complex_mix_mag.squeeze().size()

    for i in range(y):
        complex_mix_mag[0,:,i] = complex_mix_mag[0,:,i]*overall_curve

    #print(overall_curve)

    return make_wav(complex_mix_mag.squeeze(),complex_mix_phase.squeeze())

wav_out = eq("D:/songs_headphones/acappella/ghostbusters.wav",0,0,0,0)

torchaudio.save("output_0" + ".wav",wav_out.squeeze(),fs)
