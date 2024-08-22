import matplotlib.pyplot as plt
import struct
import numpy as np
from scipy import fft
from scipy import signal

def main():
    samps = 1000

    g = []
    with open('out.bin', 'rb') as fd:
        fd.seek(0, 2)
        print(fd.tell())
        fd.seek(0)
        while True:
            data = fd.read(samps * 8)
            try:
                v = np.ndarray(samps, np.float64, data)
            except TypeError:
                break
            v = np.array(v)
            #v = (v - np.min(v)) / (np.max(v) - np.min(v))
            g.append(v)
    
    gm = np.mean(g, axis=0)

    g -= gm

    rm = np.mean(g, axis=1)
    for i in range(g.shape[0]):
        g[i] -= rm[i]

    plt.imshow(g, aspect='auto')
    plt.show()

    #plt.plot(np.mean(g, axis=0))
    #plt.show()

if __name__ == '__main__':
    main()