import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
class CSA:
    def __init__(self, fitness, pop=10, Var=100, Maxit=1000, pa=0.2, beta=1.5, showplot=False, showdisp=False ):
        '''
        Parameter:
        - fitness: cost function
        - pop: population size
        - Var: number of variables
        - Maxit: maximum number of iterations
        - pa: probability parameter for random walk
        - beta: power law index for the Lévy distribution
        - showplot: display a plot of best cost vs iterations (True/False)
        - Showdisp: display message of progress during calculation (True/False)
        '''
        self.fitness = fitness
        self.pop = pop
        self.Var = Var
        self.Maxit = Maxit
        self.pa = pa
        self.beta = beta
        self.showplot = showplot
        self.showdisp = showdisp

        self.GlobalBest_cost = math.inf
        self.GlobalBest_position= np.empty((1,Var))
        self.position = np.random.randint(5000,size=(pop,Var))
        self.cost = np.empty((pop,1))
        self.bestcost = np.zeros((self.Maxit,1))
        self.k = np.empty((self.pop,self.Var))
        # Calculating sigma
        self.sigma = (math.gamma(1+self.beta)*math.sin(math.pi*self.beta/2)/
                    (math.gamma((1+self.beta)/2)*self.beta*2**((self.beta-1)/2)))**(1/self.beta)
        
    # Initialization
    def Initialization(self):
        for i in range(self.pop):
            self.cost[i] = self.fitness(self.position[i,:])
            if self.cost[i] < self.GlobalBest_cost:
                self.GlobalBest_cost = self.cost[i]
                self.GlobalBest_position = self.position[i]

    # Update position, cost and global best after perform    
    def update_position(self, n, new_cost, new_position):
        self.new_cost = new_cost
        self.new_position = new_position
        if self.new_cost < self.cost[n]:
            self.cost[n] = self.new_cost
            self.position[n]= self.new_position
            if self.cost[n] < self.GlobalBest_cost:
                self.GlobalBest_cost = self.cost[n]
                self.GlobalBest_position = self.position[n,:]
        return self.GlobalBest_cost, self.GlobalBest_position
    
    # Perform a Lévy Flight
    def levy(self, n):
        u = np.random.randn(self.Var)*self.sigma
        v = np.random.randn(self.Var)
        step = u/(abs(v)**(1/self.beta))
        c = 0.01*(self.position[n,:] - self.GlobalBest_position)*step
        new_position_levy = self.position[n,:] + c*np.random.randn(self.Var)
        new_cost_levy = self.fitness(new_position_levy)
        self.update_position(n, new_cost_levy, new_position_levy)

    # Perform a RandomWalk
    def randomwalk(self, n):       
        self.k[n,:] = np.random.random(self.Var)<self.pa
        location = np.arange(self.pop)
        np.random.shuffle(location)
        d1 = location[1]
        d2 = location[2]
        if d1 == n:
            d1 = location[3]
        if d2 == n:
            d2 = location[4]
        stepsizeK = np.random.rand(1)*(self.position[d1,:]-self.position[d2,:])
        new_position_randomwalk = self.position[n,:] + stepsizeK*self.k[n,:]
        new_cost_randomwalk = self.fitness(new_position_randomwalk)
        self.update_position(n, new_cost_randomwalk, new_position_randomwalk)

    # Plot the result
    def plot(self):
        plt.plot(self.bestcost)
        plt.xlabel('Iterations')
        plt.ylabel('Best Cost')
        plt.title('Cuckoo Search Algorithm')
        plt.show()

    # MAIN OF CUCKOO SEARCH ALGORITHM
    def Main(self):  
        self.Initialization()     
        for it in range(self.Maxit):
            for n in range (self.pop):
                # Levy Flight
                self.levy(n)
                # RandomWalk
                self.randomwalk(n)
            self.bestcost[it] = self.GlobalBest_cost

            if self.showdisp == True:
                print(f"Iteration {it}: {self.bestcost[it]}")
        if self.showplot == True:
            self.plot()

    # REFERENCE #####################################################################
    # [1] X. YANG AND SUASH DEB, "CUCKOO SEARCH VIA LÉVY FLIGHTS,"                  #                                                       #
    #     2009 WORLD CONGRESS ON NATURE & BIOLOGICALLY INSPIRED COMPUTING (NABIC),  #                                                       #
    #     2009, PP. 210-214, DOI: 10.1109/NABIC.2009.5393690.                       #
    # ###############################################################################

# Fitness 
def fitness(x):
    return sum(x**2)     

# Example Usage 
model = CSA(fitness, Var=2, pa=0.3,  pop=10, Maxit=1000, showdisp=True, showplot=True)
model.Main()
print(model.GlobalBest_position)