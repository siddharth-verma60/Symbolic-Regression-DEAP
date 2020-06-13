import operator
import math
import random
from deap import base
from deap import creator
from deap import tools
from deap import gp
from load_dataset import loadDataset
from output_writer import write_rule_population, write_predictions, write_test_accuracy_estimate
from Predicton import accEstimate, accEstimateDiscrete
import os


class DEAP_EA:

    def __init__(self, datasetNum, popSize):
        self.toolbox = base.Toolbox()
        self.fileName = str(datasetNum) + "-" + "Regression(X^4+X^3+X^2+X+1)"
        self.dataEntry = loadDataset(self.fileName + "_Train")
        self.isDiscrete = True

        self.initializeTreeProperties()
        self.population = self.toolbox.population(n=popSize)

        outputPath = os.getcwd() + "/" + self.fileName.split('-')[1] + "-Local_Output/"
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)

    def initializeTreeProperties(self):
        # Define new functions
        def protectedDiv(left, right):
            try:
                return left / right
            except ZeroDivisionError:
                return 1

        numAttributes = len(self.dataEntry[0]) - 1

        pset = gp.PrimitiveSet("MAIN", numAttributes)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protectedDiv, 2)
        pset.addPrimitive(operator.neg, 1)
        pset.addPrimitive(math.cos, 1)
        pset.addPrimitive(math.sin, 1)
        [pset.addEphemeralConstant(str(random.random()) + "eph" + str(i + 6), lambda: i) for i in range(-6, 6)]

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)

        if self.isDiscrete:
            self.toolbox.register("evaluate", self.evalSymbRegDiscrete, self.dataEntry)
        else:
            self.toolbox.register("evaluate", self.evalSymbReg, self.dataEntry)

        self.toolbox.register("select", tools.selTournament, tournsize=20)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=pset)

        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    def evalSymbReg(self, points, individual):
        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)

        # Evaluate the mean squared error between the expression and the phenotype
        sqerrors = [(func(*x[:-1]) - x[-1]) ** 2 for x in points]
        return 1 / (1 + math.sqrt(math.fsum(sqerrors) / (len(points) - 1))),

    def evalSymbRegDiscrete(self, points, individual):
        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)

        # Evaluate the balanced accuracy
        truePos = 0
        trueNeg = 0
        falsePos = 0
        falseNeg = 0
        for x in points:
            predictedPhenotype = 1 if func(*x[:-1]) > 0 else 0
            truePhenotype = int(x[-1])
            if predictedPhenotype is 1 and truePhenotype is 1:
                truePos += 1
            if predictedPhenotype is 1 and truePhenotype is 0:
                falsePos += 1
            if predictedPhenotype is 0 and truePhenotype is 0:
                trueNeg += 1
            if predictedPhenotype is 0 and truePhenotype is 1:
                falseNeg += 1

        balancedAcc = (truePos / (truePos + falseNeg) + trueNeg / (trueNeg + falsePos)) / 2
        return balancedAcc,

    def varAnd(self, population, cxpb, mutpb):
        """Part of an evolutionary algorithm applying only the variation part
        (crossover **and** mutation). The modified individuals have their
        fitness invalidated. The individuals are cloned so returned population is
        independent of the input population.
        """
        offspring = [self.toolbox.clone(ind) for ind in population]

        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1],
                                                                   offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = self.toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        return offspring

    def eaSimple(self, cxpb, mutpb, ngen, tracking_frequency):
        """This algorithm reproduce the simplest evolutionary algorithm.
        """

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = self.toolbox.select(self.population, len(self.population))

            # Vary the pool of individuals
            offspring = self.varAnd(offspring, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace the current population by the offspring
            self.population[:] = offspring

            if gen % tracking_frequency == 0:
                write_rule_population(self.fileName, self.population, gen)
                if not self.isDiscrete:
                    print(
                        "Generation: " + str(gen) + " AccuracyEstimate: " + str(
                            accEstimate(self.toolbox, self.dataEntry, self.population)))
                else:
                    print(
                        "Generation: " + str(gen) + " AccuracyEstimate: " + str(
                            accEstimateDiscrete(self.toolbox, self.dataEntry, self.population)))

        write_predictions(self.toolbox, self.fileName, self.population, ngen, self.isDiscrete)
        write_test_accuracy_estimate(self.toolbox, self.fileName, self.population, self.isDiscrete)

        return self.population


def main():
    random.seed(318)

    for i in range(1, 2):
        standaloneGP = DEAP_EA(datasetNum=i, popSize=300)
        standaloneGP.eaSimple(0.8, 0.2, ngen=30, tracking_frequency=25)
        del standaloneGP

if __name__ == "__main__":
    main()
    print("Training complete")

