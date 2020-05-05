import pandas as pd
import numpy as np


class GenerateExample:
    def __init__(self, n_items, times, seed=None):        
        self.n_items = n_items
        self.times = times
        self.seed = seed
    
    def demand(self, low, high):
        np.random.seed(self.seed)

        demands = np.random.randint(low, high, self.n_items)
        rnd_matrix = np.random.rand(self.n_items, self.times)
        normalized = rnd_matrix / rnd_matrix.sum(axis=1).reshape((-1,1))
        
        demand = np.int32(normalized * demands.reshape((-1,1)))
        self.demand = pd.DataFrame(demand)
    
        return self.demand
    
    def ordering_cost(self, sw=[30, 50], sr=[0.1, 0.3]):
        np.random.seed(self.seed)
        sw_min, sw_max = sw
        sr_min, sr_max = sr

        sw = np.zeros(self.n_items)
        sr = np.zeros(self.n_items)

        for i in range(self.n_items):
            sw[i] = np.random.randint(sw_min, sw_max)
            sr[i] = np.random.randint(sw[i] * sr_min, sw[i] * sr_max)

        return np.int32(sw), np.int32(sr)
    
    def inventory_cost(self, hw=[0.5, 3.0], hr=[1.2, 2.0]):
        np.random.seed(self.seed)
        hw_min, hw_max = hw
        hr_min, hr_max = hr

        hw = np.zeros(self.n_items)
        hr = np.zeros(self.n_items)

        for i in range(self.n_items):
            hw[i] = (hw_max - hw_min) * np.random.random() + hw_min
            hr[i] = (hr_max*hw[i] - hr_min*hw[i]) * np.random.random() + hr_min*hw[i]

        return np.round(hw, 2), np.round(hr, 2)
    
    def price_break(self, ki=[3, 2, 3]):        
        q = {}
        for i, k in zip(range(self.n_items), ki):
            lower = np.array([0, 1/4, 1/3, 1/2])
            lower = lower * self.demand.sum(1)[i]
            lower = np.round(lower, -1)[:k].astype(np.int32)
            
            upper = np.append(lower[1:k]-1, self.demand.sum().sum())
            
            q[i] = list(zip(lower, upper))
            
        p = []
        for i in ki:
            interval = np.array([1, 0.9, 0.85, 0.80])
            prices = interval[:i] * (3 - 1) * np.random.random() + 1
            p.append(list(prices.round(2)))
        
        return q, p