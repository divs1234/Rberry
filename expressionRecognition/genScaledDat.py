import pandas as pd
import numpy as np


data = pd.read_csv('fer2013.csv')
data = data['pixels']
data = [ dat.split() for dat in data]
data = np.array(data)
data = data.astype('float64')
data = [[np.divide(d,255.0) for d in dat] for dat in data]

np.save('data/Scaled.bin.npy',data)
