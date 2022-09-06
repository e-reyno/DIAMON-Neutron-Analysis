import matplotlib.pyplot as plt
import numpy as np
import diamon_read_data as dia


def plot_spect(data):
    
    plt.xscale("log")
    plt.step(data[0], data[1])
    plt.legend()
    plt.show()

def plot_detector_counts(rate_data):

    # add cumulative time
    rate_data['time'] = rate_data['Dt(s)'].cumsum()
    # plot counts over time
    plt.step(rate_data["time"], rate_data["Det1"])
    plt.xlabel("Time (s)")
    plt.ylabel("Counts")
    plt.show()
    
def background_subtraction(data,background):
    
    return data - background

def remove_times(data, time_range):
    
    return 0

def average_data():
    return 0

def neuron_energy_dist():
    return 0