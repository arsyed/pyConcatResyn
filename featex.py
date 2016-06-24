from __future__ import division
from features import mfcc
# from bob.ap import Ceps
from scipy.io import wavfile
from numpy.matlib import repmat
import numpy as np
import utils


def load_mel_chunks_from_wav(wav_file, frames_per_chunk, zero_pad=0):
    fs, sig = wavfile.read(wav_file)
    # scale to [-1, 1]; conforms to matlab wavread
    sig = sig / np.iinfo(sig.dtype).max
    if len(sig.shape) > 1:
        # mean of channels
        sig = np.mean(sig, axis=1)

    # params used in matlab, but missing from features lib:
    #   nbands: 22, fbtype: fcmel, dcttype: 1, usecmp: 1, dither: 1
    wintime = 0.032
    hoptime = 0.016
    win = np.round(fs * wintime)
    hop = np.round(fs * hoptime)
    samples_per_chunk = int((frames_per_chunk - 1) * hop + win);

    if zero_pad == 1:
        z = np.zeros(int(hop * (frames_per_chunk - 1) / 2))
        sig = np.concatenate([z, sig, z])

    nfilt = 22
    nceps = 20
    maxfreq = 8000
    preemph = 0
    with_energy = False

    # ceps = Ceps(fs, 
    #             win_length_ms=wintime*1e3,
    #             win_shift_ms=hoptime*1e3,
    #             n_filters=nfilt,
    #             n_ceps=nceps,
    #             f_max=maxfreq,
    #             pre_emphasis_coeff=preemph,
    #             mel_scale=True)
    # ceps.with_energy = with_energy
    # mix_aspc = ceps(sig * 3.3752)

    # CHECK: 3.3752?
    mix_aspc = mfcc(sig * 3.3752, 
                    fs, 
                    highfreq=maxfreq,
                    numcep=nceps,
                    winlen=wintime,
                    winstep=hoptime,
                    preemph=preemph,
                    appendEnergy=with_energy)

    assert np.all(np.isfinite(mix_aspc))
    mix_aspc = mix_aspc.T
    mix_aspc = utils.mag2db(mix_aspc)
    assert np.all(np.isfinite(mix_aspc))

    if zero_pad == 2:
        reps = int((frames_per_chunk - 1) / 2)
        mix_aspc = np.concatenate([
                     repmat(mix_aspc[:, 0][:, np.newaxis], 1, reps),
                     mix_aspc,
                     repmat(mix_aspc[:, -1], 1, reps)
                   ], axis=1)
        # also zero pad signal for wav chunks
        z = np.zeros(int(hops * (frames_per_chunk - 1) / 2))
        sig = np.concatenate([z, sig, z])
    assert np.all(np.isfinite(mix_aspc))

    n_chunks = mix_aspc.shape[1] - frames_per_chunk + 1
    chunks = np.zeros((n_chunks, frames_per_chunk * mix_aspc.shape[0]))
    wav_chunks = np.zeros((n_chunks, samples_per_chunk))
    for i in xrange(n_chunks):
        chunks[i, :] = mix_aspc[:, i:(i+frames_per_chunk)].flatten(order='F')

    return chunks, wav_chunks, sig, fs, hop, win


