cimport cython
import numpy as np
from .model import JointReplenishment




class GeneticAlgorithm(JointReplenishment):
    def __init__(self):
        pass


    def _lot_size(demand, gene):
        idx2one = []
        for i, v in enumerate(list(gene) + [1]):
            if v == 1:
                idx2one.append(i)

        result = np.zeros(12)

        for i,j in zip(idx2one[:len(gene)-1], idx2one[1:]):
            result[i] = np.sum(demand[i:j])
        return result

    def _inventory(de, re):
        """ 현재고 = 이전재고 + 공급 - 수요
        """
        result = np.zeros((12))
        for i, v in enumerate(re):
            if i == 0:
                result[i] = 0 + re[i] - de[i]
            else:
                result[i] = result[i-1] + re[i] - de[i]
        return result

    def fitness(pop):
        gene = pop.reshape((2,-1))[1]
        res1 = np.zeros((3,12))
        inv1 = np.zeros((3,12))
        for i in range(3):
            res1[i] = lot_size(d[i], gene.reshape(3,-1)[i])
            inv1[i] = inventory(d[i], res1[i])

        hrc = np.sum(inv1.sum(axis=1) * hr)
        mrc = np.sum(np.where(res > 0, 1, 0).sum(axis=1) * sr)

        gene = pop.reshape((2,-1))[0]
        res2 = np.zeros((3,12))
        inv2 = np.zeros((3,12))

        for i in range(3):
            res2[i] = lot_size(res1[i], gene.reshape(3,-1)[i])
            inv2[i] = inventory(res[i], res2[i])

        mc = np.where(res2.sum(axis=0) > 0, 1, 0).sum() * 310
        hwc = np.sum(inv2.sum(axis=1) * hw)
        mwc = np.sum(np.where(res2 > 0, 1, 0).sum(axis=1) * sw)

        value = hrc + mrc + mc + hwc + mwc

        p1 = np.where(inv1 < 0, inv1, 0).sum()
        p2 = np.where(inv2 < 0, inv2, 0).sum()

        f_value = value - (p1 + p2) * 10

        return f_value
    
    def _solve(self):



        pop = []

        for i in range(200):
            pop.append(np.random.randint(2, size=(2,36)))

        # pop.append(g1)

        best_value = np.inf

        for gg in range(1000):
            cal_fit = []
            for i in range(len(pop)):
                cal_fit.append((i, fitness(pop[i])))
                current_value = cal_fit[i][1]
                if best_value > current_value:
                    best_value = current_value

            new_pop = []
            for i in sorted(cal_fit, key=lambda x: x[1])[:40]:
                new_pop.append(pop[i[0]])

            for cross in range(20):
                rand_sel_parent1 = np.random.choice(18)
                rand_sel_parent2 = np.random.choice(range(19,36))

                parent1 = new_pop[rand_sel_parent1].copy()
                parent2 = new_pop[rand_sel_parent2].copy()

                parent1_1 = parent1[0]
                parent1_2 = parent1[1]

                parent2_1 = parent2[0]
                parent2_2 = parent2[1]

                parent1_1[:18] = parent2_1[18:]
                parent1_2[:18] = parent2_2[18:]

                child = np.zeros((2,36))
                child[0] = parent1_1
                child[1] = parent1_2

                new_pop.append(child)

            for new in new_pop[10:20]:
                rnd_i = np.random.choice(34)
                rnd_j = np.random.choice(range(rnd_i+1, 36))

                new[0][rnd_j] = new[0][rnd_i]
                new[1][rnd_i] = new[1][rnd_j]

            for i in range(60):
                new_pop.append(np.random.randint(2, size=(2,36)))

            pop = new_pop.copy()
            if gg % 100 == 0:
                print('Gene {}: best {}'.format(gg, best_value))
                
                
# example
        
        
        
        
d = np.array([
    [90, 70, 50, 40, 40, 110, 110, 0, 110, 120, 50, 10],
    [70, 30, 150, 50, 220, 150, 120, 260, 100, 70, 160, 240],
    [40, 30, 110, 200, 100, 400, 140, 80, 80, 190, 220, 140]
])

s = 310
sw = np.array([139, 90, 51])
sr = np.array([9, 37, 29])
hw = np.array([0.1, 0.1, 0.1])
hr = np.array([0.2, 0.1, 0.2])

g = np.array([
    [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
])

g = np.array(
    [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
)

g1 = np.array([
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
])