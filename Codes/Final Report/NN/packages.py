import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CUDA device configuration