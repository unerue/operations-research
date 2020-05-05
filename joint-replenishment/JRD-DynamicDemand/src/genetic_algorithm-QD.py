import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms


class GeneticAlgorithm:
    def __init__(self, d, s, sw, sr, hw, hr, q, p):
        self.d = d
        self.n_items = d.shape[0]
        self.times = d.shape[1]
        
        self.s = s
        self.sw = sw
        self.sr = sr
        
        self.hw = hw
        self.hr = hr

        self.q = q
        self.p = p

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
        
        return result.astype(np.int32)

    def _inventory(self, d, i, q):
        """Inventory
        d : demand
        i : item
        q : quantity
        """
        result = np.zeros(self.times)
        for j, v in enumerate(q):
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
        
        # Compute quantity discounts
        pcost = 0
        for i, (v1, v2) in enumerate(zip(fw_res, self.q.values())):
            for j, v3 in enumerate(v2):
                if (v1[(v1 >= v3[0]) & (v1 <= v3[1])]).sum():
                    pcost += (v1[(v1 >= v3[0]) & (v1 <= v3[1])]).sum() * self.p[i][j]            

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
        
        tc = (fw_major.sum() * self.s) + fr_minor_cost.sum() + fw_minor_cost.sum() \
                + fr_inv_cost.sum() + fw_inv_cost.sum() + pcost

        return tc,

    def optimize(self, n=100, ngen=10):
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register('choromosome', random.randint, 0, 1)
        toolbox.register('individual', tools.initRepeat, creator.Individual, 
                         toolbox.choromosome, n=(self.n_items * self.times * 2) - (self.n_items * 2))
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        toolbox.register('evaluate', self._fitness)
        toolbox.register('mate', tools.cxTwoPoint) # tools.cxOrdered, cxTwoPoint
        toolbox.register('mutate', tools.mutFlipBit, indpb=0.5) # tools.mutShuffleIndexes, indpb = 0.6) # tools.mutFlipBit 0.05
        toolbox.register('select', tools.selTournament, tournsize=3) # tools.selTournament, tournsize=3) # tools.selRoulette

        pop = toolbox.population(n=n)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)
        
        pop, self.log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.4, ngen=ngen, 
                                            stats=stats, halloffame=hof, verbose=True)
        

if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        n = 200
        ngen = 100
    else:
        n = int(sys.argv[1])
        ngen = int(sys.argv[2])

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

    q = {0: [(0, 269), (270, 399), (400, 10000)],
         1: [(0, 409), (410, 10000)],
         2: [(0, 579), (580, 869), (870, 10000)]}

    p = [[2, 1.8, 1.5], [2.8, 2.5], [2.2, 2.1, 1.9]]

    ga = GeneticAlgorithm(d, s, sw, sr, hw, hr, q, p)
    ga.optimize(n=n, ngen=ngen)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # fig, ax = plt.subplots(figsize=(7,5), dpi=100)
    # x = range(len(ga.log))
    # y = [i['min'] for i in ga.log]
    # best = np.min(y)

    # ax.plot(x, y, c='red', label='Fitness values ($\min$)')
    # ax.axhline(best, c='g', lw=1, ls='--', label='Best fitness: {}'.format(best))
    # ax.legend(loc='upper right', frameon=True, shadow=False, 
    #           fancybox=False, ncol=1, framealpha=1, edgecolor='black')
    
    # ax.set_xlabel('Generation')
    # ax.set_ylabel('Fitness values')
    # ax.grid(axis='y', linestyle='--')
    # plt.show()

    fig, ax = plt.subplots()
    x = range(len(ga.log))
    y = [i['min'] for i in ga.log]
    
    ln, = plt.plot([], [], c='red', label='Fitness values ($\min$)')

    def init():
        ax.set_xlim(0, 10)
        ax.set_ylim([np.min(y)*0.995, np.max(y)*1.005])
        # ax.axhline(np.min(y), c='g', lw=1, ls='--', label='Best fitness: {}'.format(np.min(y)))
        ax.grid(axis='y', linestyle='--')
        ax.set_title('Optimization')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness values')
        ax.legend(loc='upper right', frameon=True, shadow=False, 
                  fancybox=False, ncol=1, framealpha=1, edgecolor='black')
        return ln, 

    def update(i):
        # xdata.append(i)
        # ydata.append(y[i])
        ln.set_data(x[:i], y[:i])
        if i > 10:
            ax.set_xlim(0, np.max(x[:i])+5)
            # plt.axhline(np.min(y[:i]), c='g', lw=1, ls='--', label='Best fitness: {}'.format(np.min(y[:i])))
        return ln, 
        # ax.plot(x[:i], y[:i], c='red', label='Fitness values ($\min$)')
        # ax.axhline(np.min(y[1:i]), c='g', lw=1, ls='--', label='Best fitness: {}'.format(np.min(y[1:i])))
        



    # def animate(i):
    #     ax.plot(x[:i], y[:i], c='red', label='Fitness values ($\min$)')
    #     # ax.axhline(np.min(y[1:i]), c='g', lw=1, ls='--', label='Best fitness: {}'.format(np.min(y[1:i])))
    #     ax.set_title('Optimization')
    #     ax.set_xlabel('Generation')
    #     ax.set_ylabel('Fitness values')
        
    #     ax.grid(axis='y', linestyle='--')
    
    ani = FuncAnimation(fig, update, init_func=init, interval=5, frames=np.arange(ngen), repeat=True) # , blit=True
    # ani = FuncAnimation(fig, update, interval=2, frames=ngen)
    plt.show()
    # To save the animation, use e.g.
    #
    # ani.save('ga-qd.gif', fps=60, dpi=60)
    #
    # or
    #
    # from matplotlib.animation import FFMpegWriter

    # # plt.rcParams['animation.ffmpeg_path'] = '/Users/unerue/Dropbox/Papers/JRP/JRD-DynamicDemand/ffmpeg'
    # plt.rcParams['animation.ffmpeg_path'] = './ffmpeg'
    # writer = FFMpegWriter(fps=24, metadata=dict(artist='Kyungsu'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)

    

    