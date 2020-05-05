import numpy as np
from pulp import *


class RetailerDriven:
    """Retailer-driven model 

    Parameters:
    -----------
    d : List[int,...]
    s : integer
    sw : List[int,...]
    sr : 
    hw : 
    hr : 
    gurobi : bool
    save : bool
    """
    def __init__(self, d, s, sw, sr, hw, hr, gurobi=True, save=False):
        self.d = d
        self.n_items = d.shape[0]
        self.times = d.shape[1]
        self.QR = d.copy()
        
        self.s = s
        self.sw = sw
        self.sr = sr
        
        self.hw = hw
        self.hr = hr
        
        self.M = d.sum().sum()
        self.gurobi = gurobi
        self.save = save
        
    def _retailers(self):
        """subproblem
        """
        cost = 0
        for i in range(self.n_items):
            d = self.d[i, :].copy() # Copy to demands for sub problem
            
            # Define problems
            prob = LpProblem('Retailer{}'.format(i+1), LpMinimize)

            QR = LpVariable.dicts('QR', list(range(self.times)), lowBound=0, cat='Continuous')
            F = LpVariable.dicts('FR', list(range(self.times)), cat='Binary')
            IR = LpVariable.dicts('IR', list(range(self.times)), lowBound=0, cat='Continuous')

            prob += self.sr[i] * lpSum([F[t] for t in range(self.times)]) \
                    + self.hr[i] * lpSum([IR[t] for t in range(self.times)]) 

            prob += 0 + QR[0] - d[0] == IR[0]

            for t in range(1, self.times):
                prob += IR[t-1] + QR[t] - d[t] == IR[t]

            for t in range(self.times):
                prob += QR[t] <= F[t] * self.M

            if self.gurobi:
                prob.solve(solver=GUROBI(msg=False, epgap=0.0))
            else:
                prob.solve()
            
            cost += value(prob.objective)
            _QR = [QR[t].varValue for t in range(self.times)]
            self.QR[i, :] = _QR
            del prob

        return cost
        
    def _warehouse(self):
        """
        """
        idx = [(i, t) for i in range(self.n_items) for t in range(self.times)]
        
        prob = LpProblem('Retailer-driven model', LpMinimize)

        QW = LpVariable.dicts('QW', idx, lowBound=0, cat='Continuous')
        Y = LpVariable.dicts('Y', list(range(self.times)), cat='Binary')
        K = LpVariable.dicts('KW', idx, cat='Binary')
        IW = LpVariable.dicts('IW', idx, lowBound=0, cat='Continuous')

        prob += self.s * lpSum([Y[t] for t in range(self.times)]) \
                + lpSum([self.sw[i] * K[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                + lpSum([self.hw[i] * IW[i, t] for i in range(self.n_items) for t in range(self.times)])

        for i in range(self.n_items):
            prob += 0 + QW[i, 0] - self.QR[i, 0] == IW[i, 0]

        for i in range(self.n_items):
            for t in range(1, self.times):
                prob += IW[i, t-1] + QW[i, t] - self.QR[i, t] == IW[i, t]

        for t in range(self.times):
            prob += lpSum([QW[i, t] for i in range(self.n_items)]) <= Y[t] * self.M

        for i in range(self.n_items):
            for t in range(self.times):
                prob += QW[i, t] <= K[i, t] * self.M

        if self.gurobi:
            prob.solve(solver=GUROBI(msg=False, epgap=0.0))
        else:
            prob.solve()
        
        cost = value(prob.objective)
        if self.save:
            self.QW = np.zeros((self.n_items, self.times))
            for i in range(self.n_items):
                for t in range(self.times):
                    self.QW[i, t] = QW[i, t].varValue
        
        return cost

    def solve(self):
        self.retailers = self._retailers()
        self.warehouse = self._warehouse()
        self.total_cost = self.retailers + self.warehouse


class WarehouseDriven:
    """Warehouse-driven model
    """
    def __init__(self, d, s, sw, sr, hw, hr, gurobi=True, save=False):
        self.d = d.copy()
        self.n_items = d.shape[0]
        self.times = d.shape[1]
        
        self.s = s
        self.sw = sw
        self.sr = sr
        
        self.hw = hw
        self.hr = hr
                
        self.M = d.sum().sum()
        self.gurobi = gurobi   
        self.save = save
        
    def _create_problem(self):
        prob = LpProblem('Warehouse-driven model', LpMinimize)

        idx = [(i, t) for i in range(self.n_items) for t in range(self.times)]

        # Decision variables
        QW = LpVariable.dicts('QW', idx, lowBound=0, cat='Continuous')
        QR = LpVariable.dicts('QR', idx, lowBound=0, cat='Continuous')

        Y = LpVariable.dicts('K', list(range(self.times)), cat='Binary')
        K = LpVariable.dicts('F', idx, cat='Binary')
        F = LpVariable.dicts('F', idx, cat='Binary')

        IW = LpVariable.dicts('IW', idx, lowBound=0, cat='Continuous')
        IR = LpVariable.dicts('IR', idx, lowBound=0, cat='Continuous')

        # Objective function
        prob += self.s * lpSum(Y[t] for t in range(self.times)) \
                + lpSum([self.sw[i] * K[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                + lpSum([self.sr[i] * F[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                + lpSum([self.hw[i] * IW[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                + lpSum([self.hr[i] * IR[i, t] for i in range(self.n_items) for t in range(self.times)])
        
        # 기초재고식
        for i in range(self.n_items):
            prob += 0 + QW[i, 0] - QR[i, 0] == IW[i, 0]
            prob += 0 + QR[i, 0] - self.d[i, 0] == IR[i, 0]

        # 기말재고식
        for t in range(1, self.times):
            for i in range(self.n_items):
                prob += IW[i, t-1] + QW[i, t] - QR[i, t] == IW[i, t]
                prob += IR[i, t-1] + QR[i, t] - self.d[i, t] == IR[i, t]

        for t in range(self.times):
            prob += lpSum([QW[i, t] for i in range(self.n_items)]) <= Y[t] * self.M

        for i in range(self.n_items):
            for t in range(self.times):
                prob += QW[i, t] <= K[i, t] * self.M
                prob += QR[i, t] <= F[i, t] * self.M
        
        return prob, QW, QR
            
    def solve(self):
        prob, QW, QR = self._create_problem()
        
        if self.gurobi:
            prob.solve(solver=GUROBI(msg=False, epgap=0.0))
        else:
            prob.solve()

        self.total_cost = value(prob.objective)

        if self.save:
            self.QW = np.zeros((self.n_items, self.times))
            self.QR = np.zeros((self.n_items, self.times))
            for i in range(self.n_items):
                for t in range(self.times):
                    self.QW[i, t] = QW[i, t].varValue
                    self.QR[i, t] = QR[i, t].varValue


class RetailerDrivenQD:
    def __init__(self, d, s, sw, sr, hw, hr, p, q, ki, gurobi=True, save=False):
        self.d = d.copy()
        self.QR = d.copy()
        self.n_items = d.shape[0]
        self.times = d.shape[1]

        self.s = s
        self.sw = sw
        self.sr = sr
        
        self.hw = hw
        self.hr = hr
        
        self.p = p
        self.q = q
        self.ki = ki
    
        self.M = d.sum().sum()

        self.gurobi = gurobi
        self.save = save
        
    def _retailers(self):
        cost = 0
        for i in range(self.n_items):
            d = self.d[i, :].copy()  # demands for sub problem
            
            # Define problems
            prob = LpProblem('Retailer{}'.format(i+1), LpMinimize)

            QR = LpVariable.dicts('QR', list(range(self.times)), lowBound=0, cat='Continuous')
            F = LpVariable.dicts('F', list(range(self.times)), cat='Binary')
            IR = LpVariable.dicts('IR', list(range(self.times)), lowBound=0, cat='Continuous')

            prob += lpSum([self.sr[i] * F[t] for t in range(self.times)]) \
                    + lpSum([self.hr[i] * IR[t] for t in range(self.times)])

            prob += 0 + QR[0] - d[0] == IR[0]

            for t in range(1, self.times):
                prob += IR[t-1] + QR[t] - d[t] == IR[t]

            for t in range(self.times):
                prob += QR[t] <= F[t] * self.M

            if self.gurobi:
                prob.solve(solver=GUROBI(msg=False, epgap=0.0))
            else:
                prob.solve()
                   
            cost += value(prob.objective)
            _QR = [QR[t].varValue for t in range(self.times)]
            self.QR[i, :] = _QR
    
        return cost
        
    def _warehouse(self):
        prob = LpProblem('Warehouse', LpMinimize)
        
        idx1 = [(i, t) for i in range(self.n_items) for t in range(self.times)]
        idx2 = [(i, t, k) for i in range(self.n_items) for t in range(self.times) for k in range(self.ki[i])]

        # Decision variables
        QW = LpVariable.dicts('QW', idx2, lowBound=0, cat='Continuous')
        Y = LpVariable.dicts('Y', list(range(self.times)), cat='Binary')
        K = LpVariable.dicts('K', idx1, cat='Binary')
        IW = LpVariable.dicts('IW', idx1, lowBound=0, cat='Continuous')
        U = LpVariable.dicts('U', idx2, cat='Binary')

        prob += self.s * lpSum([Y[t] for t in range(self.times)]) \
                + lpSum([self.sw[i] * K[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                + lpSum([self.hw[i] * IW[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                + lpSum([self.p[i][k] * QW[i, t, k] for i in range(self.n_items) for t in range(self.times) for k in range(self.ki[i])])
        
        # Beginning inventory level
        for i in range(self.n_items):
            prob += 0 + lpSum([QW[i, 0, k] for k in range(self.ki[i])]) - self.QR[i, 0] == IW[i, 0]

        # Expected ending inventory level
        for t in range(1, self.times):
            for i in range(self.n_items):
                prob += IW[i, t-1] + lpSum([QW[i, t, k] for k in range(self.ki[i])]) - self.QR[i, t] == IW[i, t]

        for t in range(self.times):
            for i in range(self.n_items):
                prob += lpSum([QW[i, t, k] for k in range(self.ki[i])]) <= Y[t] * self.M

        for i in range(self.n_items):
            for t in range(self.times):
                prob += lpSum([QW[i, t, k] for k in range(self.ki[i])]) <= K[i, t] * self.M
                
        for i in range(self.n_items):
            for t in range(self.times):
                for k in range(self.ki[i]):
                    prob += QW[i, t, k] <= U[i, t, k] * self.M

        # Meet between quantity and demand
        prob += lpSum(QW) == self.d.sum().sum()

        for i in range(self.n_items):
            for t in range(self.times):
                for k in range(self.ki[i]):
                    prob += self.q[i][k][0] + self.M*(U[i, t, k] - 1) <= QW[i, t, k]
                    prob += self.q[i][k][1] + self.M*(1 - U[i, t, k]) >= QW[i, t, k]

        for t in range(self.times):
            for i in range(self.n_items):
                prob += lpSum([U[i, t, k] for k in range(self.ki[i])]) == 1

        if self.gurobi:
            prob.solve(solver=GUROBI(msg=False, epgap=0.0))
        else:
            prob.solve()
                
        cost = value(prob.objective)
        if self.save:
            tmp = [[[0 for t in range(self.times)] for k in range(self.ki[i])] for i in range(self.n_items)]
            for i in range(self.n_items):
                for t in range(self.times):
                    for k in range(self.ki[i]):
                        tmp[i][k][t] = QW[i, t, k].varValue

            self.QW = np.zeros((np.sum(self.ki),self.times))
            i = 0
            for ti in tmp:
                for tj in ti:
                    self.QW[i] = tj
                    i += 1

        return cost

    def solve(self):
        self.retailers = self._retailers()
        self.warehouse = self._warehouse()
        self.total_cost = self.retailers + self.warehouse


class WarehouseDrivenQD:
    def __init__(self, d, s, sw, sr, hw, hr, p, q, ki, gurobi=True, save=False):
        self.d = d.copy()
        self.n_items = d.shape[0]
        self.times = d.shape[1]

        self.s = s
        self.sw = sw
        self.sr = sr
        
        self.hw = hw
        self.hr = hr
        
        self.p = p
        self.q = q
        self.ki = ki

        self.M = d.sum().sum()
        self.gurobi = gurobi
        self.save = save
        
    def _create(self):
        prob = LpProblem('Warehouse-driven model', LpMinimize)

        idx1 = [(i, t) for i in range(self.n_items) for t in range(self.times)]
        idx2 = [(i, t, k) for i in range(self.n_items) for t in range(self.times) for k in range(self.ki[i])]

        # Decision variables
        QW = LpVariable.dicts('QW', idx2, lowBound=0, cat='Continuous')
        QR = LpVariable.dicts('QR', idx1, lowBound=0, cat='Continuous')

        Y = LpVariable.dicts('Y', list(range(self.times)), cat='Binary')
        K = LpVariable.dicts('K', idx1, cat='Binary')
        F = LpVariable.dicts('F', idx1, cat='Binary')

        IW = LpVariable.dicts('IW', idx1, lowBound=0, cat='Continuous')
        IR = LpVariable.dicts('IR', idx1, lowBound=0, cat='Continuous')

        U = LpVariable.dicts('U', idx2, cat='Binary')

        # Objective function
        prob += self.s * lpSum([Y[t] for t in range(self.times)]) \
                + lpSum([self.sw[i] * K[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                + lpSum([self.sr[i] * F[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                + lpSum([self.hw[i] * IW[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                + lpSum([self.hr[i] * IR[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                + lpSum([self.p[i][k] * QW[i, t, k] for i in range(self.n_items) for t in range(self.times) for k in range(self.ki[i])])
        
        # Beginning inventory level
        for i in range(self.n_items):
            prob += 0 + lpSum([QW[i, 0, k] for k in range(self.ki[i])]) - QR[i, 0] == IW[i, 0]
            prob += 0 + QR[i, 0] - self.d[i, 0] == IR[i, 0]

        # Expected ending inventory level
        for t in range(1, self.times):
            for i in range(self.n_items):
                prob += IW[i, t-1] + lpSum([QW[i, t, k] for k in range(self.ki[i])]) - QR[i, t] == IW[i, t]
                prob += IR[i, t-1] + QR[i, t] - self.d[i, t] == IR[i, t]

        for t in range(self.times):
            for i in range(self.n_items):
                prob += lpSum([QW[i, t, k] for k in range(self.ki[i])]) <= Y[t] * self.M

        for i in range(self.n_items):
            for t in range(self.times):
                prob += lpSum([QW[i, t, k] for k in range(self.ki[i])]) <= K[i, t] * self.M
                prob += QR[i, t] <= F[i, t] * self.M
                
        for i in range(self.n_items):
            for t in range(self.times):
                for k in range(self.ki[i]):
                    prob += QW[i, t, k] <= U[i, t, k] * self.M

        for i in range(self.n_items):
            for t in range(self.times):
                for k in range(self.ki[i]):
                    prob += self.q[i][k][0] + self.M*(U[i, t, k] - 1) <= QW[i, t, k]
                    prob += self.q[i][k][1] + self.M*(1 - U[i, t, k]) >= QW[i, t, k]

        for t in range(self.times):
            for i in range(self.n_items):
                prob += lpSum([U[i, t, k] for k in range(self.ki[i])]) == 1

        # Meet between quantity and demand
        prob += lpSum(QW) == self.d.sum().sum()
        
        return prob, QW, QR

    def solve(self):
        prob, QW, QR = self._create()

        if self.gurobi:
            prob.solve(solver=GUROBI(msg=False, epgap=0.0))
        else:
            prob.solve()
             
        self.total_cost = value(prob.objective)

        if self.save:
            tmp = [[[0 for t in range(self.times)] for k in range(self.ki[i])] for i in range(self.n_items)]
            for i in range(self.n_items):
                for t in range(self.times):
                    for k in range(self.ki[i]):
                        tmp[i][k][t] = QW[i, t, k].varValue

            self.QW = np.zeros((np.sum(self.ki),self.times))
            i = 0
            for ti in tmp:
                for tj in ti:
                    self.QW[i] = tj
                    i += 1
        
            self.QR = np.zeros((self.n_items, self.times))
            for i in range(self.n_items):
                for t in range(self.times):
                    self.QR[i, t] = QR[i, t].varValue

