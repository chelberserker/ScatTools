import numpy as np
import matplotlib.pyplot as plt
import h5py

class D11_SANS():

    def __init__(self, filename, data=True):
        self.q = np.array([])
        self.I = np.array([])
        self.I_err = np.array([])
        self.q_err = np.array([])
        self.q_fit = np.array([])
        self.I_fit = np.array([])
        self.sample_name = ''
        if data:
            self.load_data(filename)
        else:
            self.load_fit(filename)

    def __repr__(self):
        return self.sample_name

    def get_sample_name(self):
        return self.sample_name

    def load_data(self, filename):
        with open(filename, 'r') as f:
            a = f.read()
            self.sample_name = a.split('Subtitle:')[1].split('\n')[0].replace(' ', '')
            data = np.loadtxt(filename, skiprows=53).T
            self.q = data[0] * 10
            self.I = data[1]
            self.I_err = data[2]
            self.q_err = data[3] * 10

    def plot(self, ax=None, bckg=0.0, label=None):
        if not ax:
            fig, ax = plt.subplots(1, 1)
        else:
            pass
        if not label:
            label = self.sample_name
        else:
            label = label

        if isinstance(bckg, float):
            ax.errorbar(self.q, self.I - bckg, yerr=self.I_err, label=label, fmt='.', capsize=3, alpha=0.25)
        else:
            ax.errorbar(self.q, self.I - bckg.I, yerr=self.I_err, label=label, fmt='.', capsize=3, alpha=0.25)
        ax.set_xlabel('$q, \\mathrm{nm}^{-1}$')
        ax.set_ylabel('$\\mathrm{\\Delta}\\Sigma / \\mathrm{\\Delta}\\Omega, \\mathrm{cm}^{-1}$')
        ax.loglog()
        ax.legend()


class ID02_SAXS():
    def __init__(self, filename, data=True, sample_thickness=0.2):
        self.q = np.array([])
        self.I = np.array([])
        self.I_err = np.array([])
        self.q_fit = np.array([])
        self.I_fit = np.array([])
        self.sample_name = ''
        self.sample_thickness = sample_thickness   # for 2 mm capillary to convert into 1/cm
        if data:
            self.load_data(filename)
        else:
            self.load_fit(filename)

    def __repr__(self):
        return self.sample_name

    def load_fit(self, filename):
        fit = np.loadtxt(filename).T
        # print(fit)
        self.q_fit = fit[0]
        self.I_fit = fit[1]

    def load_data(self, filename):
        with h5py.File(filename, 'r') as file:
            self.I = np.array(file['entry_0000']['saxsutilities']['data']['I'])[0] / self.sample_thickness
            self.I_err = np.array(file['entry_0000']['saxsutilities']['data']['Idev'])[0] / self.sample_thickness
            self.q = np.array(file['entry_0000']['saxsutilities']['data']['q'])
            self.sample_name = str(file['entry_0000']['saxsutilities']['data']['header_array']['title'][()]).split("\'")[1]

    def plot(self, ax=None, norm_factor=1, label=None):
        if not ax:
            fig, ax = plt.subplots(1, 1)
        else:
            pass
        if not label:
            label = self.sample_name
        else:
            label = label

        ax.errorbar(self.q, self.I / norm_factor, self.I_err / norm_factor, fmt='o', capsize=2, linewidth=1,
                    markersize=1, label=label)
        ax.set_xlabel('$q, \\mathrm{nm}^{-1}$')
        ax.set_ylabel('$\\mathrm{Intensity, cm}^{-1}$')
        ax.legend()
        ax.loglog()
