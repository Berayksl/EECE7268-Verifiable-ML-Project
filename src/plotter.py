#SCRIPT TO PLOT THE TRAINING REWARDS FROM .NPY FILES FOR COMPARISON
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import tkinter as tk
from tkinter import filedialog
import sys

def select_plot_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

    file_path = filedialog.askopenfilename(
        title="Select Plot File",
        initialdir=model_dir,
        filetypes=[("NP Files", "*.npy"), ("All Files", "*.*")])
    return file_path


def plot_data(data_list):
    colors = ['b', 'green', 'orange']
    labels = [r"SAC with $\Phi_1$", r"SAC with $\Phi_2$", "SAC (baseline)"]
    fig, ax = plt.subplots()
    for i, data in enumerate(data_list):
        mean = data.mean(axis=0)
        std = data.std(axis=0, ddof=1)  # sample std
        lo = mean - 2 * std
        hi = mean + 2 * std

        x = np.arange(0, data.shape[1])
        ax.plot(x, mean, linewidth=2, color=colors[i], label=labels[i])
        ax.fill_between(x, lo, hi, alpha=0.2, color=colors[i])
        
    ax.set_xlim(0, max(d.shape[1] for d in data_list) - 1)
    formatter = ScalarFormatter(useMathText=True)   # use 10^4 style
    formatter.set_powerlimits((-1, 1))              # force scientific notation when appropriate
    formatter.set_scientific(True)
    ax.yaxis.set_major_formatter(formatter)

    # show only 1 decimal digit
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
    ax.yaxis.get_offset_text().set_fontsize(10)     # adjust font if needed
    ax.tick_params(axis='y', labelsize=10)

    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.legend()

    # Save the plot
    file_name = "comparison_plot.png"
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')
    plot_dir = os.path.join(base_dir, file_name)
    plt.tight_layout()
    plt.savefig(plot_dir, dpi = 200) #save the plot with the current date and time
    print(f"Plot saved to {plot_dir}")
    plt.show()


if __name__ == "__main__":
    data_list = []
    for i in range(3):
        print(f"Select plot file for dataset {i+1}")
        plot_file = select_plot_file()
        if not plot_file:
            print("No file selected. Exiting.")
            sys.exit()
        data = np.load(plot_file)
        data_list.append(data)

    #plot the comparison graph
    plot_data(data_list)

