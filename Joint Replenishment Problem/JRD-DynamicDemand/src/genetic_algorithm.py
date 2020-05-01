import random
import time
import argparse
import numpy as np
import multiprocessing
from deap import base
from deap import creator
from deap import tools
from deap import algorithms


class GeneticAlgorithm:
    def __init__(self, d, s, sw, sr, hw, hr):
        self.d = d
        self.n_items = d.shape[0]
        self.times = d.shape[1]
        
        self.s = s
        self.sw = sw
        self.sr = sr
        
        self.hw = hw
        self.hr = hr

    def _decoding(self, d, i, individual):
        individual = np.array(individual)
        dummies = np.zeros(self.times)
        dummies[-1] = d[i][-1]
        
        for j in range(len(individual)-2, -1, -1):
            if individual[j+1] == 1:
                value = d[i][j]
            else:
                value = d[i][j] + dummies[j+1]

            dummies[j] = value
        
        result = individual * dummies
        
        return result

    def _inventory(self, d, i, q):
        result = np.zeros(self.times)
        for j in range(len(q)):
            if j == 0:
                result[j] = 0 + q[j] - d[i][j]
            else:
                result[j] = result[j-1] + q[j] - d[i][j]
        return result

    def _fitness(self, individual):
        individual = np.array(individual).reshape((self.n_items * 2),-1)
        fr = individual[:int((self.n_items * 2) / 2)]
        fw = individual[int((self.n_items * 2) / 2):]
        
        fr = np.insert(fr, 0, 1, axis=1)
        fw = np.insert(fw, 0, 1, axis=1)
        
        fr_res = np.zeros((fr.shape[0], fr.shape[1]))
        fw_res = np.zeros((fw.shape[0], fw.shape[1]))
            
        # Decision Quantity
        for i, ind in enumerate(fr):
            value = self._decoding(self.d, i, ind)
            fr_res[i] = value
            
        for i, ind in enumerate(fw):
            value = self._decoding(fr_res, i, ind)
            fw_res[i] = value
        
        # Minor ordering costs
        fr_minor = np.where(fr_res > 0, 1, 0)
        fw_minor = np.where(fw_res > 0, 1, 0)
        fr_minor_cost = fr_minor.sum(1) * self.sr
        fw_minor_cost = fw_minor.sum(1) * self.sw
        
        # Major ordering cost
        fw_major = (fw_minor == 1).any(axis=0)
        
        # Calculate ending inventory
        fr_inv = np.zeros((fr.shape[0], fr.shape[1]))
        fw_inv = np.zeros((fw.shape[0], fw.shape[1]))
            
        for i, ind in enumerate(fr_res):
            value = self._inventory(self.d, i, ind)
            fr_inv[i] = value
        
        for i, ind in enumerate(fw_res):
            value = self._inventory(fr_res, i, ind)
            fw_inv[i] = value
        
        fr_inv_cost = fr_inv.sum(1) * self.hr
        fw_inv_cost = fw_inv.sum(1) * self.hw
        
        tc = (fw_major.sum() * self.s) + fr_minor_cost.sum() + fw_minor_cost.sum() + fr_inv_cost.sum() + fw_inv_cost.sum()

        return tc,

    def optimize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('n', type=int, help='population', default=100)
        parser.add_argument('ngen', type=int, help='iteration', default=50)
        parser.add_argument('max_ngen', type=int, help='maximum ngen without improvement')

        args = parser.parse_args()
        n = args.n
        ngen = args.ngen
        max_ngen = args.max_ngen

        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Multiprocessing
        pool = multiprocessing.Pool()
        toolbox.register('map', pool.map)

        toolbox.register('choromosome', random.randint, 0, 1)
        toolbox.register('individual', tools.initRepeat, creator.Individual, 
                         toolbox.choromosome, n=(self.n_items * self.times * 2) - (self.n_items * 2))
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        toolbox.register('evaluate', self._fitness)
        toolbox.register('mate', tools.cxTwoPoint) # tools.cxOrdered
        toolbox.register('mutate', tools.mutFlipBit, indpb=0.5) # tools.mutShuffleIndexes, indpb = 0.6) # tools.mutFlipBit
        toolbox.register('select', tools.selTournament, tournsize=3) # tools.selTournament, tournsize=3) # tools.selRoulette

        # Evaluate the individuals with an invalid fitness
        pop = toolbox.population(n=n)
        CXPB, MUTPB = 0.6, 0.4

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)

        self.logbook = tools.Logbook()
        self.logbook.header = 'gen', 'evals', 'std', 'min', 'avg', 'max'

        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # self.log = {'min': [], 'max': [], 'avg': [], 'std': []}
        fits = [ind.fitness.values[0] for ind in pop]
        g = 0
        mg = 1
        best_fitness = np.inf
        start_time = time.time()
        while g < ngen:
            g = g + 1
            
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # print('  Evaluated {} individuals'.format(len(invalid_ind)))

            record = stats.compile(pop)
            self.logbook.record(gen=g, evals=len(invalid_ind), **record)
            print(self.logbook.stream)

            pop[:] = offspring
            fits = [ind.fitness.values[0] for ind in pop]
            
            # Termination rule
            best_ind = tools.selBest(pop, 1)[0]
            current_fitness = best_ind.fitness.values[0]
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                mg = 1

            # Termination rule 
            if mg % max_ngen == 0:
                print('-- Maximum ngen without improvement -- {}'.format(time.time() - start_time))
                break
            
            mg += 1

        print('-- End of (successful) evolution -- {}'.format(time.time() - start_time))
                
        best_ind = tools.selBest(pop, 1)[0]
        print('Best individual\'s fitness {}'.format(best_ind.fitness.values))
        

        # pop, self.log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.4, ngen=ngen, 
        #                                     stats=stats, halloffame=hof, verbose=True)
        

if __name__ == '__main__':
    d = np.array([
        [90, 70, 50, 40, 40, 110, 110, 0, 110, 120, 50, 10],
        [70, 30, 150, 50, 220, 150, 120, 260, 100, 70, 160, 240],
        [40, 30, 110, 200, 100, 400, 140, 80, 80, 190, 220, 140]
    ])
    s = 100
    sw = np.array([45, 40, 43])
    sr = np.array([17, 19, 11])
    hw = np.array([0.1, 0.1, 0.1])
    hr = np.array([0.2, 0.1, 0.2])

    ga = GeneticAlgorithm(d, s, sw, sr, hw, hr)
    ga.optimize()

    # Draw chart
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(dpi=200)

    x = range(len(ga.logbook))
    y = [i['min'] for i in ga.logbook]
    best = np.min(y)

    ax.plot(x, y, c='red', label='Fitness values ($\min$)')
    ax.axhline(best, c='g', lw=1, ls='--', label='Best fitness value: {}'.format(best))

    ax.legend(loc='upper right', frameon=True, shadow=False, 
              fancybox=False, ncol=1, framealpha=1, edgecolor='black')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness values')
    ax.grid(axis='y', linestyle='--')
    plt.show()

