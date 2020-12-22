# Data Processing
import numpy as np
import pandas as pd

# Model Library
import tensorflow as tf

# Parallel Compute
import torch 
import torch.nn as nn

# Data Visualization
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Utility Libraries
import random
import math
from time import time
from copy import deepcopy
from datetime import datetime

# Initialize Device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print("Torch Version\t", torch.__version__)
print("Using Device\t", torch.cuda.get_device_name(0))