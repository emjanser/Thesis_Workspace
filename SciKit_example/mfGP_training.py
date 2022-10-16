from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import matplotlib
import numpy as np
import random as rnd

import time
import os

# Commited

from matplotlib import pyplot as plt

matplotlib.use("PDF")

np.random.seed(rnd.randint(0, 1000000)) # makes it possible for .permutation() to be randomized every single run of the code

""" Emrecan Serin 11/10/22 -
Comment 1: I have added a random integer number generator into np.random.seed() to make sure LF and HF points changed everytime the code is run.

Comment 2: I have realised that "L1mean" was a column array and ".hstack" function on line 63 was trying to stack a row (X_hf) with a column (L1mean). 
So I added .reshape(-1,1) on line 58 to turn "L1mean" into a row array to fix the array dimension error.

Comment 3: I have had problems on lines 83, 91 and 101 with ".fill_between" function from matplotlib, I was getting "too many indices for array" error,
probably because of how I played with the code to make it work or some other reason. As a solution, I have tried multiple things that came to my mind for 
hours but could not fix it. I commented them out in the end and the plot seems to be working fine.

Comment 4: I had a familiar array dimension problem with line 94 so I used .reshape(-1,1) again to fix it.
"""
start_timer = time.time()
def hf(x): # defines the hf and lf sine functions
    return 1.8*np.sin(8.0*np.pi*x)*2*x  # the accurately simulated sin model for HF data


def lf(x):
    return np.sin(8.0*np.pi*x)*x # the inaccurately simulated sin model for LF data


# X = np.linspace(-np.pi, np.pi, 1000)[:, np.newaxis]
X = np.linspace(0, 1, 1000)[:,np.newaxis] # Increases the dimension of the array from 1D to 2D to 3D... Also turns it into a row vector if it's from 1D to 2D

Nhf=10 # Using the linspace with pi, a lot more points were needed to get an accurate model.
Nlf=50

#sample LF and hF model randomly
X_lf = np.random.permutation(X)[0:Nlf] # basically randomly aranges the 50 different values in X array, from 0 to 50 [0:Nlf(50)]  "print(len(X_lf)) = 50"
X_hf = np.random.permutation(X_lf)[0:Nhf]

gpr_hf = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=200).fit(X_hf,hf(X_hf)) # Gaussian Regressor by sklearn
gpr_lf = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=200).fit(X_lf,lf(X_lf)) #.fit learns the correlation betwen the two values

# MF start point
L1mean = gpr_lf.predict(X_hf) # using the gpr_lf model which uses X_lf points and X_lf data,
                              # it predicts an output using the inputs of X_hf points

L1mean = L1mean.reshape(-1,1)

L2_train = np.hstack((X_hf, L1mean))

gpr_mf_l2 = GaussianProcessRegressor(kernel=RBF()*RBF()+RBF(),n_restarts_optimizer=200).fit(L2_train,hf(X_hf))


#Plotting -- 

fig, axs = plt.subplots(4)
axs[0].plot(X,hf(X),label="High Fidelity / Exact") # main result line we want to match with
axs[0].plot(X_lf, lf(X_lf),'bo', label="Low fidelity samples") # low fidelity dots
axs[0].plot(X_hf, hf(X_hf),'ro', label="High fidelity samples") # high fidelity dots
axs[0].legend(bbox_to_anchor=(0.9, 1), loc='upper left', fontsize='x-small')

pred_hf_mean, pred_hf_std = gpr_hf.predict(X, return_std=True)

axs[1].plot(X,hf(X),label="High Fidelity / Exact")
axs[1].plot(X, pred_hf_mean, 'k', lw=3, label="GP mean (trained on red dots)")
axs[1].plot(X_hf, hf(X_hf),'ro', label="High fidelity samples")
# axs[1].fill_between(X[:,0], pred_hf_mean[:,0]-pred_hf_std, pred_hf_mean[:,0]+pred_hf_std,alpha=0.2, color='k', label="+/- 1 std")
axs[1].legend(bbox_to_anchor=(0.9, 1), loc='upper left', fontsize='x-small')

pred_lf_mean, pred_lf_std = gpr_lf.predict(X, return_std=True)

axs[2].plot(X,hf(X),label="High Fidelity / Exact")
axs[2].plot(X, pred_lf_mean, 'k', lw=3, label="GP mean (trained on blue dots)")
axs[2].plot(X_lf, lf(X_lf),'bo', label="Low fidelity samples")
# axs[2].fill_between(X[:,0], pred_lf_mean[:,0]-2*pred_lf_std, pred_lf_mean[:,0]+2*pred_lf_std,alpha=0.2, color='k', label="+/- 2 std")
axs[2].legend(bbox_to_anchor=(0.9, 1), loc='upper left', fontsize='x-small')

pred_lf_mean = pred_lf_mean.reshape(-1,1)

L2_test = np.hstack((X, pred_lf_mean))
pred_mf_mean, pred_mf_std = gpr_mf_l2.predict(L2_test, return_std=True)

axs[3].plot(X,hf(X),label="High Fidelity / Exact")
axs[3].plot(X, pred_mf_mean, 'k', lw=3, label="Deep GP mean (trained on all dots)")
# axs[3].fill_between(X[:,0], pred_mf_mean[:,0]-2*pred_mf_std, pred_mf_mean[:,0]+2*pred_mf_std,alpha=0.2, color='k', label="+/- 2 std")
axs[3].legend(bbox_to_anchor=(0.9, 1), loc='upper left', fontsize='x-small')

fig.text(0.5, 0.03, '$x$', ha='center')
fig.text(0.03, 0.5, '$y=f(x)$', va='center', rotation='vertical')

abs_path = os.path.abspath(__file__)
dir = os.path.dirname(abs_path)

plt.savefig(dir + "\output_plot.pdf")

print(f"Finished in {(time.time() - start_timer)/60} minutes." )