import pandas as pd
import numpy as np


class GenerateExample:
    """Generate a random numerical example for constant 
    """
    def __init__(self, n_items: int, seed: int = None): 
        self.n_items = n_items
        self.seed = seed
    

    def demand(self, low: int, high: int):
        np.random.seed(self.seed)
        return np.random.randint(low, high, self.n_items).tolist()
    

    def ordering_cost(self, sw_range=[30, 50], sr_range=[0.1, 0.3]):
        np.random.seed(self.seed)
        sw_min, sw_max = sw_range
        sr_min, sr_max = sr_range

        swi = np.zeros(self.n_items)
        sri = np.zeros(self.n_items)
        for i in range(self.n_items):
            swi[i] = np.random.randint(sw_min, sw_max)
            sri[i] = np.random.randint(swi[i] * sr_min, swi[i] * sr_max)

        return np.int32(swi).tolist(), np.int32(sri).tolist()
    

    def inventory_cost(self, hw_range=[0.5, 3.0], hr_range=[1.2, 2.0]):
        np.random.seed(self.seed)
        hw_min, hw_max = hw_range
        hr_min, hr_max = hr_range

        hwi = np.zeros(self.n_items)
        hri = np.zeros(self.n_items)
        for i in range(self.n_items):
            hwi[i] = (hw_max - hw_min) * np.random.random() + hw_min
            hri[i] = (hr_max*hwi[i] - hr_min*hwi[i]) * np.random.random() + hr_min*hwi[i]

        return np.round(hwi, 2).tolist(), np.round(hri, 2).tolist()
    

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