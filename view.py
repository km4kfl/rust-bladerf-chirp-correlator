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
    with open(args.data, 'rb') as fd:
        fd.seek(0, 2)
        print(fd.tell())
        fd.seek(0)
        while True:
            data = fd.read((samps + 1) * 4)
            try:
                v = np.ndarray(samps, np.float32, data)
            except TypeError:
                break
            v = np.array(v)
            g.append((v[0], v[1:]))
            print(v[0])
            x += 1

    #g.sort(key=lambda i: i[0])
    f = [v[0] for v in g]
    g = [v[1] for v in g]

    g = np.array(g)

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
    
    #print('subtracting column average')
    #gm = np.mean(g, axis=0)
    #g -= gm

    #g = np.abs(g) ** 0.1

    print('doing gaussian filter')
    #g = ndimage.gaussian_filter(g, (args.gy, args.gx))

    #gm = np.mean(g, axis=0)
    #g -= gm

    print('plotting')
    sol = 299792458

    d = sol / args.sps * g.shape[1] * 0.5

    plt.title('Correlation')
    plt.ylabel('time')
    plt.xlabel('approx. meters')
    plt.imshow(g, aspect='auto', extent=(0, d, g.shape[0], 0), cmap='rainbow')
    #plt.imshow(g, aspect='auto', cmap='rainbow')
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