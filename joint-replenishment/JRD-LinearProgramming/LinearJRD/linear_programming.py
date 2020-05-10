import re
import numpy as np
from pulp import *
from tabulate import tabulate


class LinearProgramming:
    """ Linear Programming

    Parameters:
    -----------
    di : int
        - demand for item i
    """
    def __init__(self, di, sw, swi, hwi, sri, hri, m=10, n_ks=10, n_fs=10, gurobi=True, verbose=True):
        self.di = di
        self.sw = sw
        self.swi = swi
        self.hwi = hwi
        self.sri = sri
        self.hri = hri
        self.n_items = len(di)
        self.m = m
        self.n_ks = n_ks
        self.n_fs = n_fs
        self.gurobi = gurobi
        self.verbose = verbose
            

    def _tminmax(self):
        # tmax
        numerator = np.sum([self.swi[i] for i in range(self.n_items)]) 
        denominator = np.sum([self.di[i]*self.hwi[i] for i in range(self.n_items)])
        tmax = np.sqrt((2*(self.sw+numerator)) / denominator)
        
        # tmin
        tmin = np.min([np.sqrt((2*self.swi[i]) / (self.di[i]*self.hwi[i])) for i in range(self.n_items)])
        return tmin, tmax


    def _linear_programming(self, t):
        indexs = [(i, k, f) for i in range(self.n_items) for k in range(self.n_ks) for f in range(self.n_fs)]
        prob = LpProblem('Linear Programming', LpMinimize)
        p = LpVariable.dicts('p', indexs, lowBound=0, cat='Binary')
        
        prob += (self.sw/t) + lpSum([(self.swi[i]/t) * lpSum([(p[i, k, f]*(1/(k+1))) for k in range(self.n_ks) for f in range(self.n_fs)]) for i in range(self.n_items)])  \
                + (t * lpSum([(self.di[i]*self.hwi[i]) * lpSum([((((f+1)-1)*(k+1)) / (2*(f+1)))*p[i, k, f] for k in range(self.n_ks) for f in range(self.n_fs)]) for i in range(self.n_items)])) \
                + lpSum([(self.sri[i]/t) * lpSum([((f+1)/(k+1))*p[i, k, f] for k in range(self.n_ks) for f in range(self.n_fs)]) for i in range(self.n_items)]) \
                + (t * lpSum([(self.di[i]*self.hri[i]) * lpSum([((k+1) / (2*(f+1)))*p[i, k, f] for k in range(self.n_ks) for f in range(self.n_fs)]) for i in range(self.n_items)]))

        for i in range(self.n_items):
            prob += lpSum([p[i, k, f] for k in range(self.n_ks) for f in range(self.n_fs)]) == 1

        if self.gurobi:
            prob.solve(solver=GUROBI(msg=False, epgap=0.0))
        else:
            prob.solve()
        
        tc = value(prob.objective)
        tmp = []
        for v in prob.variables():
            if v.varValue == 1:
                tmp.append(list(map(int, re.findall('\d+', v.name))))
                
        tmp = np.array(tmp)
        tmp = tmp[tmp[:, 0].argsort()]
        ks = list(tmp[:, 1] + 1)
        fs = list(tmp[:, 2] + 1)

        return tc, ks, fs


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


    def optimize(self):
        headers = ['j', 'r', 'Tj', 'T(r)', 'T(r-1)', 'ki', 'fi', 'TCj', 'LP']
        table = []
        
        tmin, tmax = self._tminmax()
        m = 10
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
                lp, ks, fs = self._linear_programming(t)
                ki.append(ks)
                fi.append(fs)

                # Step 7      
                t = self._find_t(ki[r], fi[r])
                before_t = self._find_t(ki[r-1], fi[r-1]) # ERROR?!
                # before_t = self._find_t(ki[r], fi[r-1]) # ERROR?!
                total = self._tc(t, ki[r], fi[r])

                table.append([j, r, ts[j-1], t, before_t, ki[r], fi[r], np.round(total, 2), np.round(lp, 2)])
                
                # Step 8 
                if t != before_t:
                    # Go to Step 4
                    r = r + 1
                
                else:
                    table.append([j, r, ts[j-1], t, before_t, ki[r], fi[r], '{:.2f}*'.format(total), np.round(lp, 2)])
                    
                    if total < self.best_total:
                        self.best_total = total
                        self.best_ki = ki[r]
                        self.best_fi = fi[r]
                        best_solution = [[j, r, ts[j-1], t, before_t, ki[r], fi[r], np.round(self.best_total, 2), lp]]
                    
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
    
    lp = LinearProgramming(di, sw, swi, hwi, sri, hri, m=10, n_ks=10, n_fs=10, gurobi=False, verbose=True)
    lp.optimize()
