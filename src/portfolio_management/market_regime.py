import pandas as pd
import numpy as np
import numpy.typing as npt

# from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tqdm import tqdm
