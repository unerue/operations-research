import numpy as np
from tabulate import tabulate


class RANDAlgorithm:
    def __init__(self, di, sw, swi, hwi, sri, hri, m=10, verbose=True):
        """ RAND Algorithm

        Parameters:
        -----------
        di : demand for item i
        """
        self.di = di
        self.sw = sw
        self.swi = swi
        self.hwi = hwi
        self.sri = sri
        self.hri = hri
        self.n_items = len(di)
        self.m = m
        self.verbose = verbose
            
    def _tminmax(self):
        # tmax
        numerator = np.sum([self.swi[i] for i in range(self.n_items)]) 
        denominator = np.sum([self.di[i]*self.hwi[i] for i in range(self.n_items)])
        tmax = np.sqrt((2*(self.sw+numerator)) / denominator)
        
        # tmin
        tmin = np.min([np.sqrt((2*self.swi[i]) / (self.di[i]*self.hwi[i])) for i in range(self.n_items)])
        return tmin, tmax

    def _find_k(self, t, i, fi):
        numerator = 2*(self.swi[i] + fi[i]*self.sri[i])
        denominator = t**2 * self.di[i] * (self.hwi[i] + ((self.hri[i] - self.hwi[i]) / fi[i]))
        if numerator / denominator > 0:
            return numerator / denominator
        else:
            return 0

    def _find_f(self, t, i, ki):
        numerator = ki[i]**2 * t**2 * self.di[i] * (self.hri[i] - self.hwi[i])
        denominator = 2 * self.sri[i]
        if numerator / denominator > 0:
            return numerator / denominator
        else:
            return 0

    def _integer_multiple_k(self, t, fi, r):
        ks = []
        for i in range(self.n_items):
            k = self._find_k(t, i, fi[r-1])
            for l in range(1, 1000):
                if l*(l-1) <= k and l*(l+1) >= k:
                    ks.append(l)
                    break
        return ks

    def _integer_multiple_f(self, t, ki, r):
        
        fs = []
        for i in range(self.n_items):
            f = self._find_f(t, i, ki[r])
            for l in range(1, 1000):
                if l*(l-1) <= f and l*(l+1) >= f:
                    fs.append(l)
                    break
    
        assert len(fs) == self.n_items
        return fs

    def _find_t(self, ki, fi):
        numerator = 2 * (self.sw + np.sum([(self.swi[i]+fi[i]*self.sri[i]) / ki[i] for i in range(self.n_items)]))
        denominator = np.sum([ki[i]*self.di[i]*(self.hwi[i]+((self.hri[i] - self.hwi[i]) / fi[i])) for i in range(self.n_items)])
        return np.round(np.sqrt(numerator / denominator), 4)

    def _tc(self, t, ki, fi):
        rs1 = (self.sw + np.sum([self.swi[i] / ki[i] for i in range(self.n_items)])) / t
        rs2 = np.sum([((fi[i]-1)*ki[i]*t*self.di[i]*self.hwi[i]) / (2*fi[i]) for i in range(self.n_items)])
        rs3 = np.sum([(fi[i]*self.sri[i]) / (ki[i]*t) for i in range(self.n_items)])
        rs4 = np.sum([(ki[i]*t*self.di[i]*self.hri[i]) / (2*fi[i]) for i in range(self.n_items)])
        return rs1 + rs2 + rs3 + rs4


    # @output   
    def optimize(self):
        headers = ['j', 'r', 'Tj', 'T(r)', 'T(r-1)', 'ki', 'fi', 'TCj']
        table = []
        
        tmin, tmax = self._tminmax()
        ts = np.linspace(tmin, tmax, num=self.m).tolist()
        j = 1
        
        self.best_total = np.inf
        self.best_ki = None
        self.best_fi = None
        best_solution = None
        while True:
            r = 0
            fi = np.ones((1, self.n_items), dtype=np.int32).tolist()
            ki = np.ones((1, self.n_items), dtype=np.int32).tolist()
                        
            # Step 4
            r = 1
            t = ts[j-1]
            while True:
                # Step 5
                ks = self._integer_multiple_k(t, fi, r)
                ki.append(ks)

                fs = self._integer_multiple_f(t, ki, r)
                fi.append(fs)

                # Step 7      
                t = self._find_t(ki[r], fi[r])
                before_t = self._find_t(ki[r-1], fi[r-1]) # ERROR?!
                # before_t = self._find_t(ki[r], fi[r-1]) # ERROR?!
                total = self._tc(t, ki[r], fi[r])

                table.append([j, r, ts[j-1], t, before_t, ki[r], fi[r], np.round(total, 2)])
                
                # Step 8 
                if t != before_t:
                    # Go to Step 4
                    r = r + 1
                
                else:
                    table.append([j, r, ts[j-1], t, before_t, ki[r], fi[r], '{:.2f}*'.format(total)])
                    
                    if total < self.best_total:
                        self.best_total = total
                        self.best_ki = ki[r]
                        self.best_fi = fi[r]
                        best_solution = [[j, r, ts[j-1], t, before_t, ki[r], fi[r], np.round(self.best_total, 2)]]
                    
                    # Step 9
                    # Go to Step 3
                    if j != self.m+1:
                        break
            
            j = j + 1
            if j == self.m+1:
                break
        
        self.log = table
        if self.verbose:
            print(tabulate(table, headers=headers))
            
        print('\nBest Solution:')
        print(tabulate(best_solution, headers=headers))

if __name__ == "__main__":
    di = [2907, 4973, 1640, 3964, 1693, 4890, 2109, 1871, 2902, 582]
    sw = 100
    swi = [34, 38, 35, 43, 30, 35, 34, 45, 41, 39]
    hwi = [2.9, 2.6, 2.5, 1.1, 0.6, 1.3, 1.2, 1.3, 2.9, 2.8]
    sri = [3.7, 11.7, 8.4, 4.8, 13.5, 9.2, 12, 9.7, 14.4, 9.3]
    hri = [2.6, 3.7, 2.1, 2.7, 3.9, 1.0, 3.4, 3.8, 3.2, 5.9]
    
    rand = RANDAlgorithm(di, sw, swi, hwi, sri, hri, m=10, verbose=True)
    rand.optimize()
