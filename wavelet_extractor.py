import numpy as np
import segyio


def process_seismic_data(filename):
    """
    Process seismic data from a SEGY file and estimate the wavelet.

    Parameters:
    filename (str): The path to the SEGY file.

    Returns:
    tuple: A tuple containing the wavelet time axis, estimated wavelet, wavelet spectrum, and seismic data array.
    """
    f = segyio.open(filename, ignore_geometry=True)

    traces = segyio.collect(f.trace)[:]
    ntraces, nt = traces.shape

    t = f.samples
    il = f.attributes(189)[:]
    xl = f.attributes(193)[:]

    # Define regular IL and XL axes
    il_unique = np.unique(il)
    xl_unique = np.unique(xl)

    il_min, il_max = min(il_unique), max(il_unique)
    xl_min, xl_max = min(xl_unique), max(xl_unique)

    dt = t[1] - t[0]
    dil = min(np.unique(np.diff(il_unique)))
    dxl = min(np.unique(np.diff(xl_unique)))

    ilines = np.arange(il_min, il_max + dil, dil)
    xlines = np.arange(xl_min, xl_max + dxl, dxl)
    nil, nxl = ilines.size, xlines.size

    ilgrid, xlgrid = np.meshgrid(np.arange(nil), np.arange(nxl), indexing="ij")

    # Look-up table
    traces_indeces = np.full((nil, nxl), np.nan)
    iils = (il - il_min) // dil
    ixls = (xl - xl_min) // dxl
    traces_indeces[iils, ixls] = np.arange(ntraces)
    traces_available = np.logical_not(np.isnan(traces_indeces))

    # Reorganize traces in regular grid
    d = np.zeros((nil, nxl, nt))
    d[
        ilgrid.ravel()[traces_available.ravel()],
        xlgrid.ravel()[traces_available.ravel()],
    ] = traces

    nil, nxl, nt = len(ilines), len(xlines), len(t)

    # Estimar a wavelet
    t_wav = np.arange(16) * (dt / 1000)
    t_wav = np.concatenate((np.flipud(-t_wav[1:]), t_wav), axis=0)

    wav_est_fft = np.mean(
        np.abs(
            np.fft.fft(d[::2, ::2, int(2500 // dt) : int(3500 // dt)], 2**8, axis=-1)
        ),
        axis=(0, 1),
    )
    fwest = np.fft.fftfreq(2**8, d=dt / 1000)

    wav_est = np.real(np.fft.ifft(wav_est_fft)[:16])
    wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
    wav_est = wav_est / wav_est.max()

    return t_wav, wav_est, fwest, wav_est_fft, d


# Exemplo de uso
filename = "data/volve.sgy"
t_wav, wav_est, fwest, wav_est_fft, d = process_seismic_data(filename)

output_data = np.column_stack((t_wav, wav_est))
np.savetxt('wavelet_generic.txt', output_data, delimiter=',', header="t_wav, wav_est")