from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from andi_datasets.models_phenom import models_phenom
import csv
# auxiliaries
import numpy as np
import matplotlib.pyplot as plt
