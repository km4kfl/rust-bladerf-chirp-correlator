"""This program reads the output from the Rust program and displays it in MatPlotLib. It also
cleans up the data with smoothing and filtering.
"""
import argparse
import matplotlib.pyplot as plt
import struct
import numpy as np
from scipy import fft
from scipy import signal
from scipy import ndimage

def main(args):
    samps = args.points

    print('loading')

    x = 0
    g = []
    m = []
    with open(args.data, 'rb') as fd:
        fd.seek(0, 2)
        print(fd.tell())
        fd.seek(0)
        while True:
            data = fd.read(samps * 2 * 4)
            try:
                v = np.ndarray(samps * 2, np.float32, data)
            except TypeError:
                break
            g.append(v[0::2])
            m.append(v[1::2])
            x += 1

    g = np.array(g)
    m = np.array(m)

    #g = g[2:, :]

    #g = g[28::32, :]
    
    #plt.scatter(np.linspace(0, g.shape[0], g.shape[0]), g[:, 0])
    #plt.show()

    #plt.imshow(g)
    #plt.show()
    #exit()
    
    print('processing')

    #h = []
    #for y in range(g.shape[0]):
    #    v = g[y, :]
    #    v = v[1:] - v[:-1]
    #    h.append(v)
    #g = np.array(h)

    #g[g < 0] = np.nan

    #g = np.abs(g)

    #print('subtracting column average')
    #gm = np.mean(g, axis=0)
    #g -= gm

    #g = np.abs(g) ** 0.1

    print('doing gaussian filter')

    #gm = np.mean(g, axis=0)
    #g -= gm

    g = ndimage.gaussian_filter(g, (args.gy, args.gx))

    print('plotting')
    sol = 299792458

    d = sol / args.sps * g.shape[1] * 0.5

    ff = 1.83e9
    g = (g - 250) / 250 * 5e3 + ff

    # g = c / (c + vs) * ff 
    # g / ff = c / (c + vs) [+]
    # ff / g = (c + vs) / c [+]
    # ff / g = (c + vs) * (1 / c) [+]
    # ff / g = (1 / c) * c + (1 / c) * vs [+]
    # ff / g - (1 / c) * c = (1 / c) * vs [+]
    # (ff / g - (1 / c) * c) / (1 / c) = vs [+]

    # convert to meters/second for doppler shift
    g = (ff / g - (1 / sol) * sol) / (1 / sol) 

    # convert to mph
    g = g / 1600 * 60 * 60

    plt.title('Correlation')
    plt.ylabel('time')
    plt.xlabel('approx. meters')
    plt.imshow(g, aspect='auto', extent=(0, d, g.shape[0], 0), cmap='rainbow')
    #plt.imshow(g, aspect='auto', cmap='rainbow')
    plt.show()

    #plt.plot(g[:, 0])
    #plt.show()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    _help = 'Anything below this gets filtered out.'
    ap.add_argument('--mag-threshold', type=float, default=270, help=_help)
    _help = 'Samples per second.'
    ap.add_argument('--sps', type=float, default=520834, help=_help)
    _help = 'Gaussian filter Y axis sigma.'
    ap.add_argument('--gx', type=float, default=32, help=_help)
    _help = 'Gaussian filter X axis sigma.'
    ap.add_argument('--gy', type=float, default=16, help=_help)
    _help = 'Post threshold. Anything higher gets filtered out.'
    ap.add_argument('--post-threshold', type=float, default=0.4, help=_help)
    _help = 'The data file path.'
    ap.add_argument('--data', type=str, default='out.bin', help=_help)
    _help = 'The number of points per scan aka sample_distance.'
    ap.add_argument('--points', type=int, default=200, help=_help)
    main(ap.parse_args())