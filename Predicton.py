import math


def getPrediction(toolbox, dataPoint, population):
    sum_prediction = 0
    for tree in population:
        func = toolbox.compile(expr=tree)
        sum_prediction += func(*dataPoint)

    return sum_prediction / len(population)


def getBestPrediction(toolbox, dataPoint, population):
    best_tree = population[0]
    best_fitness = best_tree.fitness.values[0]

    for tree in population:
        if tree.fitness.values[0] > best_fitness:
            best_tree = tree
            best_fitness = tree.fitness.values[0]

    func = toolbox.compile(expr=best_tree)
    return func(*dataPoint)


def accEstimate(toolbox, dataEntry, population):
    squaredErrorSum = 0
    for dataPoint in dataEntry:
        predictedPhenotype = getPrediction(toolbox, dataPoint[:-1], population)
        squaredErrorSum += (predictedPhenotype - dataPoint[-1]) ** 2

    RMSE = math.sqrt(squaredErrorSum / len(dataEntry))
    accuracy = 1 / (1 + RMSE)
    return [accuracy, RMSE]


def bestAccEstimate(toolbox, dataEntry, population):
    squaredErrorSum = 0
    for dataPoint in dataEntry:
        bestPredictedPhenotype = getBestPrediction(toolbox, dataPoint[:-1], population)
        squaredErrorSum += (bestPredictedPhenotype - dataPoint[-1]) ** 2

    RMSE = math.sqrt(squaredErrorSum / len(dataEntry))
    accuracy = 1 / (1 + RMSE)
    return [accuracy, RMSE]


def getPredictionDiscrete(toolbox, dataPoint, population):
    one_vote = 0
    zero_vote = 0
    for tree in population:
        func = toolbox.compile(expr=tree)
        predictedPhenotype = func(*dataPoint)
        if predictedPhenotype > 0:
            one_vote += 1
        else:
            zero_vote += 1

    return 1 if one_vote > zero_vote else 0


def getBestPredictionDiscrete(toolbox, dataPoint, population):
    best_tree = population[0]
    best_fitness = best_tree.fitness.values[0]

    for tree in population:
        if tree.fitness.values[0] > best_fitness:
            best_tree = tree
            best_fitness = tree.fitness.values[0]

    func = toolbox.compile(expr=best_tree)
    return 1 if func(*dataPoint) > 0 else 0


def accEstimateDiscrete(toolbox, dataEntry, population):
    # Evaluate the balanced accuracy
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    for x in dataEntry:
        predictedPhenotype = getPredictionDiscrete(toolbox, x[:-1], population)
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
    return [balancedAcc, 0]  # 0 because RMSE=0 for discrete phenotype


def bestAccEstimateDiscrete(toolbox, dataEntry, population):
    # Evaluate the balanced accuracy
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    for x in dataEntry:
        bestPredictedPhenotype = getBestPredictionDiscrete(toolbox, x[:-1], population)
        truePhenotype = int(x[-1])
        if bestPredictedPhenotype is 1 and truePhenotype is 1:
            truePos += 1
        if bestPredictedPhenotype is 1 and truePhenotype is 0:
            falsePos += 1
        if bestPredictedPhenotype is 0 and truePhenotype is 0:
            trueNeg += 1
        if bestPredictedPhenotype is 0 and truePhenotype is 1:
            falseNeg += 1

    balancedAcc = (truePos / (truePos + falseNeg) + trueNeg / (trueNeg + falsePos)) / 2
    return [balancedAcc, 0]  # 0 because RMSE=0 for discrete phenotype
