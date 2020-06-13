import sys

def loadData(dataFile):
    """ Load the data file. Check for and ignore any rows/instances that are missing an endpoint/phenotype/class value.
    Detect and store basic dataset parameters (i.e. number of attributes and number of instances)."""
    print("DataManagement: Loading Data... " + str(dataFile))
    try:
        datasetList = []
        labelPhenotype = 'Class'
        f = open(dataFile, newline='')

        trainHeaderList = f.readline().rstrip('\r\n').split('\t')  # strip off first row

        # Phenotype column identification ------------------------------------------------------------------------------------------------------------------
        phenotypeRef = trainHeaderList.index(labelPhenotype)
        print("DataManagement: Phenotype Column Location = " + str(phenotypeRef))

        numAttributes = len(trainHeaderList) - 1
        print("DataManagement: Number of Attributes = " + str(numAttributes))

        # Grab the remaining dataset rows
        for line in f:
            lineList = line.strip('\r').split('\t')
            lineList = [float(i) for i in lineList]
            datasetList.append(lineList)
        f.close()

        numTrainInstances = len(datasetList)
        print("DataManagement: Number of Instances = " + str(numTrainInstances))  # DEBUG

    except IOError as xxx_todo_changeme:
        (errno, strerror) = xxx_todo_changeme.args
        print("Could not Read File!")
        print(("I/O error(%s): %s" % (errno, strerror)))
        raise
    except ValueError:
        print("Could not convert data to an integer.")
        raise
    except:
        print(("Unexpected error:", sys.exc_info()[0]))
        raise
    return datasetList


def loadDataset(fileName):
    folderName=fileName.split('_')[0].split('-')[1]
    path = "/Users/siddharthverma/Documents/Ryan-Research/DEAP_Symbolic_Regression/"+folderName+"-Datasets/"
    # path = "/Users/siddharthverma/Documents/Ryan-Research/DEAP_Symbolic_Regression/Regression(2-X1:3+5*X2)-Datasets/"
    return loadData(path + fileName + ".txt")
