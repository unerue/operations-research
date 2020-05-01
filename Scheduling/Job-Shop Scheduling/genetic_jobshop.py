import random
import pandas as pd
import numpy as np
import datetime

from deap import base
from deap import creator
from deap import tools

xw
PATH = 'data.xlsx'
sheet_name = 'Sheet1'

data = pd.read_excel(PATH, sheet_name=sheet_name)
data.set_index(['Jobs', 'Machines'], inplace=True)

jobs = np.unique(data.index.get_level_values(0))
machines = np.unique(data.index.get_level_values(1))
jobs = dict(zip(jobs, range(1,len(jobs)+1)))
machines = dict(zip(machines, range(1,len(machines)+1)))

data.rename(index=jobs, inplace=True)
data.rename(index=machines, inplace=True)

jobs = data.index.get_level_values(0)
machines = data.index.get_level_values(1)

pivoted = data.pivot_table(index='Machines', columns='Jobs', fill_value=1000)
pivoted

gene = [1, 5, 4, 2, 3, 2, 3, 1, 4, 4, 2, 3, 1]

m1 = gene[:5]
m1
m2 = gene[5:9]
m2
m3 = gene[9:]
m3

col_name = pivoted.columns.get_level_values(0)[0]
pivoted.reindex([2,3,1])
pivoted.iloc[1]
