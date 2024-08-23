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

    x = 0
    g = []
    with open(args.data, 'rb') as fd:
        fd.seek(0, 2)
        print(fd.tell())
        fd.seek(0)
        while True:
            data = fd.read((samps + 1) * 8)
            try:
                v = np.ndarray(samps, np.float64, data)
            except TypeError:
                break
            v = np.array(v)
            best_mag = v[0]
            print(best_mag)
            if best_mag > args.mag_threshold:
                v = v[1:]
                g.append(v)
            x += 1
    
    g = np.array(g)

    gm = np.mean(g, axis=0)
    g -= gm

    g0 = ndimage.gaussian_filter(g, (args.gy, args.gx))

    g = g0

    g = (g - np.min(g)) / (np.max(g) - np.min(g))

    h = []
    for y in range(g.shape[0]):
        if g[y, 0] < args.post_threshold:
            h.append(g[y, :])
    g = np.array(h)

    sol = 299792458

    d = sol / args.sps * g.shape[1] * 0.5

    plt.title('Phased Compensated Correlation Over Time')
    plt.ylabel('time')
    plt.xlabel('approx. meters')
    plt.imshow(g, aspect='auto', extent=(0, d, g.shape[0], 0))
    plt.show()

    plt.plot(g[:, 0])
    plt.show()

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