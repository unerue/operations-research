from pulp import *


class Basic:
    def __init__(self, name, n_items, times, ki):
        self.n_items = n_items
        self.times = times
        self.ki = ki
        self.prob = LpProblem(name, LpMinimize)

        idx1 = [(i, t) for i in range(self.n_items) for t in range(self.times)]
        idx2 = [(i, t, k) for i in range(self.n_items) for t in range(self.times) for k in range(self.ki[i])]

        # Decision variables
        self.QW = LpVariable.dicts('QW', idx2, lowBound=0, cat='Continuous')
        self.QR = LpVariable.dicts('QR', idx1, lowBound=0, cat='Continuous')

        self.KW = LpVariable.dicts('KW', list(range(self.times)), cat='Binary')
        self.FW = LpVariable.dicts('FW', idx1, cat='Binary')
        self.FR = LpVariable.dicts('FR', idx1, cat='Binary')

        self.IW = LpVariable.dicts('IW', idx1, lowBound=0, cat='Continuous')
        self.IR = LpVariable.dicts('IR', idx1, lowBound=0, cat='Continuous')

        self.U = LpVariable.dicts('U', idx2, cat='Binary')


class IntegratedQDM(Basic):
    def __init__(self, d, s, sw, sr, hw, hr, p, q, ki, gurobi=True):
        super().__init__('Integrated Model', d.shape[0], d.shape[1], ki)
        self.d = d.copy()
        
        self.s = s
        self.sw = sw
        self.sr = sr
        
        self.hw = hw
        self.hr = hr
        
        self.p = p
        self.q = q
    
        self.M = d.sum().sum()
        self.gurobi = gurobi
        self.verbose = False
        
    def _create(self):
        # Objective function
        self.prob += self.s * lpSum([self.KW[t] for t in range(self.times)]) \
                     + lpSum([self.sw[i] * self.FW[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                     + lpSum([self.sr[i] * self.FR[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                     + lpSum([self.hw[i] * self.IW[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                     + lpSum([self.hr[i] * self.IR[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                     + lpSum([self.p[i][k] * self.QW[i, t, k] for i in range(self.n_items) for t in range(self.times) for k in range(self.ki[i])])
        
        # Beginning inventory level
        for i in range(self.n_items):
            self.prob += 0 + lpSum([self.QW[i, 0, k] for k in range(self.ki[i])]) - self.QR[i, 0] == self.IW[i, 0]
            self.prob += 0 + self.QR[i, 0] - self.d.iloc[i, 0] == self.IR[i, 0]

        # Expected ending inventory level
        for t in range(1, self.times):
            for i in range(self.n_items):
                self.prob += self.IW[i, t-1] + lpSum([self.QW[i, t, k] for k in range(self.ki[i])]) - self.QR[i, t] == self.IW[i, t]
                self.prob += self.IR[i, t-1] + self.QR[i, t] - self.d.iloc[i, t] == self.IR[i, t]

        for t in range(self.times):
            for i in range(self.n_items):
                self.prob += lpSum([self.QW[i, t, k] for k in range(self.ki[i])]) <= self.KW[t] * self.M

        for i in range(self.n_items):
            for t in range(self.times):
                self.prob += lpSum([self.QW[i, t, k] for k in range(self.ki[i])]) <= self.FW[i, t] * self.M
                self.prob += self.QR[i, t] <= self.FR[i, t] * self.M
                
        for i in range(self.n_items):
            for t in range(self.times):
                for k in range(self.ki[i]):
                    self.prob += self.QW[i, t, k] <= self.U[i, t, k] * self.M

        for i in range(self.n_items):
            for t in range(self.times):
                for k in range(self.ki[i]):
                    self.prob += self.q[i][k][0] + self.M*(self.U[i, t, k] - 1) <= self.QW[i, t, k]
                    self.prob += self.q[i][k][1] + self.M*(1 - self.U[i, t, k]) >= self.QW[i, t, k]

        for t in range(self.times):
            for i in range(self.n_items):
                self.prob += lpSum([self.U[i, t, k] for k in range(self.ki[i])]) == 1
        
        return self

    def solve(self):
        self._create()

        if self.gurobi:
            self.prob.solve(solver=GUROBI(msg=False, epgap=0.0))
        else:
            self.prob.solve()
        
        if self.prob.status and self.verbose:
            print('Integrated model: {}'.format(LpStatus[self.prob.status]))           
        
        self.total_cost = value(self.prob.objective)
        del self.prob
        

class IndividualQDM:
    def __init__(self, d, s, sw, sr, hw, hr, p, q, ki, gurobi=True):
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
        self.verbose = False
        
    def _retailers(self):
        cost = 0
        for i in range(self.n_items):
            d = self.d.iloc[i, :].copy()  # demands for sub problem
            
            # Define problems
            prob = LpProblem('Retailer{}'.format(i+1), LpMinimize)

            QR = LpVariable.dicts('QR', list(range(self.times)), lowBound=0, cat='Continuous')
            FR = LpVariable.dicts('FR', list(range(self.times)), cat='Binary')
            IR = LpVariable.dicts('IR', list(range(self.times)), lowBound=0, cat='Continuous')

            prob += lpSum([self.sr[i] * FR[t] for t in range(self.times)]) \
                    + lpSum([self.hr[i] * IR[t] for t in range(self.times)])

            prob += 0 + QR[0] - d.iloc[0] == IR[0]

            for t in range(1, self.times):
                prob += IR[t-1] + QR[t] - d.iloc[t] == IR[t]

            for t in range(self.times):
                prob += QR[t] <= FR[t] * self.M

            if self.gurobi:
                prob.solve(solver=GUROBI(msg=False, epgap=0.0))
            else:
                prob.solve()
            
            if prob.status and self.verbose:
                print('Ratailer{}: {}'.format(i+1, LpStatus[prob.status]))
        
            cost += value(prob.objective)
            _QR = [QR[t].varValue for t in range(self.times)]
            self.QR.iloc[i, :] = _QR
            del prob

        return cost
        
    def _warehouse(self):
        prob = LpProblem('Warehouse', LpMinimize)
        
        idx1 = [(i, t) for i in range(self.n_items) for t in range(self.times)]
        idx2 = [(i, t, k) for i in range(self.n_items) for t in range(self.times) for k in range(self.ki[i])]

        # Decision variables
        QW = LpVariable.dicts('QW', idx2, lowBound=0, cat='Continuous')
        KW = LpVariable.dicts('KW', list(range(self.times)), cat='Binary')
        FW = LpVariable.dicts('FW', idx1, cat='Binary')
        IW = LpVariable.dicts('IW', idx1, lowBound=0, cat='Continuous')
        U = LpVariable.dicts('U', idx2, cat='Binary')

        prob += self.s * lpSum([KW[t] for t in range(self.times)]) \
                + lpSum([self.sw[i] * FW[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                + lpSum([self.hw[i] * IW[i, t] for i in range(self.n_items) for t in range(self.times)]) \
                + lpSum([self.p[i][k] * QW[i, t, k] for i in range(self.n_items) for t in range(self.times) for k in range(self.ki[i])])
        
        # Beginning inventory level
        for i in range(self.n_items):
            prob += 0 + lpSum([QW[i, 0, k] for k in range(self.ki[i])]) - self.QR.iloc[i, 0] == IW[i, 0]

        # Expected ending inventory level
        for t in range(1, self.times):
            for i in range(self.n_items):
                prob += IW[i, t-1] + lpSum([QW[i, t, k] for k in range(self.ki[i])]) - self.QR.iloc[i, t] == IW[i, t]

        for t in range(self.times):
            for i in range(self.n_items):
                prob += lpSum([QW[i, t, k] for k in range(self.ki[i])]) <= KW[t] * self.M

        for i in range(self.n_items):
            for t in range(self.times):
                prob += lpSum([QW[i, t, k] for k in range(self.ki[i])]) <= FW[i, t] * self.M
                
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

        if self.gurobi:
            prob.solve(solver=GUROBI(msg=False, epgap=0.0))
        else:
            prob.solve()

        if prob.status and self.verbose:
                print('Warehouse: {}'.format(LpStatus[prob.status]))           
        
        cost = value(prob.objective)
        del prob

        return cost

    def solve(self):
        self.retailers = self._retailers()
        self.warehouse = self._warehouse()
        self.total_cost = self.retailers + self.warehouse
        if self.verbose:
            print('Done...')