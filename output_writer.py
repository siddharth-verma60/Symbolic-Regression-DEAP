import sys
import os
import deap
from load_dataset import loadDataset
from Predicton import *


def write_rule_population(fileName, population, gen):
    folderName=fileName.split('-')[1]
    outputLocation = os.getcwd() + "/" + folderName + "-Local_Output/"
    fileName = fileName + '_' + str(gen) + "_RulePop.txt"
    dataFile = outputLocation + fileName
    try:
        headerList = ['Specified', 'Tree', 'Fitness']
        f = open(dataFile, 'w')

        # Writing the headers -----------------------------------------------------------------------------------------
        for head in headerList:
            f.write(head + '\t')
        f.write('\n')

        # Grab the remaining dataset rows
        for tree in population:
            # get specified tree args
            tree_args = set()
            for idx, node in enumerate(tree):
                if isinstance(node, deap.gp.Terminal):
                    if node.name.split('G')[0] == 'AR':
                        nodeAtt = int(float(node.name.split('G')[1]))
                        tree_args.add(nodeAtt)
            specifiedAtt = list(tree_args)

            f.write(str(specifiedAtt) + '\t')
            f.write(str(tree) + '\t')
            f.write(str(tree.fitness.values[0]) + '\t')
            f.write('\n')
        f.close()

    except IOError as xxx_todo_changeme:
        (errno, strerror) = xxx_todo_changeme.args
        print("Could not Read File!")
        print(("I/O error(%s): %s" % (errno, strerror)))
        raise
    except:
        print(("Unexpected error:", sys.exc_info()[0]))
        raise


def write_predictions(toolbox, fileName, population, gen, isDiscrete):
    folderName=fileName.split('-')[1]
    outputLocation = os.getcwd() + "/" + folderName + "-Local_Output/"
    outputFileName = fileName + '_' + str(gen) + "_Predictions.txt"
    dataFile = outputLocation + outputFileName
    print("DataManagement: Writing preditions... " + fileName)
    try:
        dataEntry = loadDataset(fileName + "_Test")
        headerList = ['InstanceID', 'EndpointPrediction', 'BestPrediction', 'TrueEndpoint']
        f = open(dataFile, 'w')

        # Writing the headers -----------------------------------------------------------------------------------------
        for head in headerList:
            f.write(head + '\t')
        f.write('\n')

        # Grab the remaining dataset rows
        for id, dataPoint in enumerate(dataEntry):
            if isDiscrete:
                phenotype = getPredictionDiscrete(toolbox, dataPoint[0:-1], population)
                bestPhenotype = getBestPredictionDiscrete(toolbox, dataPoint[0:-1], population)
            else:
                phenotype = getPrediction(toolbox, dataPoint[0:-1], population)
                bestPhenotype = getBestPrediction(toolbox, dataPoint[0:-1], population)
            f.write(str(id) + '\t')
            f.write(str(phenotype) + '\t')
            f.write(str(bestPhenotype) + '\t')
            f.write(str(dataPoint[-1]) + '\t')
            f.write('\n')
        f.close()

    except IOError as xxx_todo_changeme:
        (errno, strerror) = xxx_todo_changeme.args
        print("Could not Read File!")
        print(("I/O error(%s): %s" % (errno, strerror)))
        raise
    except:
        print(("Unexpected error:", sys.exc_info()[0]))
        raise


def write_test_accuracy_estimate(toolbox, fileName, population, isDiscrete):
    folderName=fileName.split('-')[1]
    outputLocation = os.getcwd() + "/" + folderName + "-Local_Output/"
    outputFileName = fileName.split('-')[1] + "_Accuracy.txt"
    dataFile = outputLocation + outputFileName
    print("DataManagement: Writing accuracies... " + fileName)
    try:
        dataEntry = loadDataset(fileName + "_Test")

        # Writing the headers -----------------------------------------------------------------------------------------
        if not os.path.exists(dataFile):
            f = open(dataFile, 'w')
            headerList = ['S.No.', 'Accuracy', 'BestAccuracy', 'RMSE', 'BestRMSE']
            for head in headerList:
                f.write(head + '\t')
            f.write('\n')
        else:
            f = open(dataFile, 'a')

        if isDiscrete:
            accuracy = accEstimateDiscrete(toolbox, dataEntry, population)
            bestAccuracy = bestAccEstimateDiscrete(toolbox, dataEntry, population)
        else:
            accuracy = accEstimate(toolbox, dataEntry, population)
            bestAccuracy = bestAccEstimate(toolbox, dataEntry, population)

        serialNumber = fileName.split('-')[0]

        f.write(serialNumber + '\t')
        f.write(str(accuracy[0]) + '\t')
        f.write(str(bestAccuracy[0]) + '\t')
        f.write(str(accuracy[1]) + '\t')
        f.write(str(bestAccuracy[1]) + '\t')
        f.write('\n')
        f.close()

    except IOError as xxx_todo_changeme:
        (errno, strerror) = xxx_todo_changeme.args
        print("Could not Read File!")
        print(("I/O error(%s): %s" % (errno, strerror)))
        raise
    except:
        print(("Unexpected error:", sys.exc_info()[0]))
        raise
