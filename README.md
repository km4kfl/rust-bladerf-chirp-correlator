# About

This is in essence just a basic chirp correlator. It does one thing that is unique. It adjusts
the phase of the individual frequency components of the chirp as a function of time. This may or
may not provide a better signal to noise ratio.

The Python program `view.py` takes the data, does some filtering on it, and displays the result
using `matplotlib`. 

It provides some interesting results that depend on your setup. If you use omni-directional
antennas you're going to get patterns but these are reflections from a near iso-tropic source
so while interesting they might not be anything you can discern. If you use directional
antennas the reflections will primarily come and be strongest in the direction of the antenna
you can discern some features about the signal as it traveled forward.