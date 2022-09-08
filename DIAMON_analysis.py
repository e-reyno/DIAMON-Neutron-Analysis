import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import diamon_read_data as dia


def plot_spect(data):
    energy, flux = extract_spectrum(data)
    plt.xscale("log")
    plt.step(energy, flux)
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
    
def plot_dose_rate(df, x, y, plot_type, x_err=None, y_err=None):
    
    graph = df.plot(x,y, kind=plot_type, xerr=x_err, yerr=y_err)
    plt.show()
    
def background_subtraction(data,background):
    
    return data - background


def remove_times(data, time_range):
    
    return 0


def average_data():
    return 0


def neuron_energy_dist():
    return 0


def extract_spectrum(data):
    
    energy = data.energy_bins
    flux = data.flux_bins
    return energy, flux

def get_energy_range(unfold_dataseries):
    
    
    return

def get_energy_range(unfold_data):
    
    if isinstance(unfold_data, pd.DataFrame):
        
        thermal = unfold_data.thermal
        epitherm = unfold_data.epi
        fast = unfold_data.fast        
        
        return thermal, epitherm, fast
    
def fit_gaussian_spect():
    
    return 0

def find_abs_error(dataframe):
    
    for i, col in enumerate(dataframe.columns):
        if 'un%' in col:
            dataframe["abs_err " + dataframe.columns[i-1]] = dataframe[dataframe.columns[i-1]] * (dataframe[col]/100)
            
    return dataframe


def stack_bar_plot(data_frame, cols, xlabel, ylabel):
    stack_df = (data_frame.filter(cols)).astype(float)
    ax = stack_df.plot(kind='bar', stacked=True, figsize = (12,8))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    return(ax)


file_path = r"C:\Users\sfs81547\Documents\DIAMON project\DIAMON_OUT\0052\F_unfold.txt"
folder_path = r"C:\Users\sfs81547\Documents\DIAMON project\DIAMON_OUT\*"
all_data = dia.read_folder(folder_path)
out = all_data[1][0]
data = dia.read_unfold_file(file_path)
plot_spect(data)
out = find_abs_error(out)
plot_dose_rate(out, 't(s)', 'H*(10)r', 'scatter')
columns = ['Ther%',	'Epit%', 'Fast%']
stack_bar_plot(out, columns, 'index', '%')