#####Coen169_project2.py
###Sanjay Tamizharasu

import numpy as np
import math
import operator

trainData = np.loadtxt("train.txt")
test5 = np.loadtxt("test5.txt")
test10 = np.loadtxt("test10.txt")
test20 = np.loadtxt("test20.txt")



##################### COSINE SIMILARITY ##############################

def cosineSimilarity(trainSet, testSet):
    sumOfBoth = 0.0
    currSimilarity = 0.0
    sumOfTrain = 0.0
    sumOfTest = 0.0

    testUserRatings = []
    trainUserRatings = []

    for key in trainSet:
        testUserRatings.append(testSet[key])
        trainUserRatings.append(trainSet[key])

    for i in range(0,len(trainUserRatings)):
        f = trainUserRatings[i]
        b = testUserRatings[i]

        if f!= 0 and b!=0:
            sumOfBoth += float(f*b)
            sumOfTrain += float(math.pow(f,2))#(f*f)
            sumOfTest +=  float(math.pow(b,2))#(b*b)

    if(math.sqrt(sumOfTrain)*math.sqrt(sumOfTest)) == 0:
        return 0.0
    else:
        denominator = float(math.sqrt(sumOfTrain))*float(math.sqrt(sumOfTest))
        return float(sumOfBoth/denominator)

def predictRating(movieNumber, trainSet, testUserMovieRatings):
    rating = 0.0
    preliminaryCosineValues = []
    userRatingsDict = {}
    userWeightsDict = {}
    numerator = 0.0
    denominator = 0.0
    newTrainSet = []
    movieNumber -= 1 #note: movies array is 1-1000 but array is 0 indexed
    count = len(testUserMovieRatings)-1


    for i in range(0,len(trainSet)):
        zeroCount = 0
        #creating new array with training users that rated the movie we want to predict and rated the movies our test user rated
        if trainSet[i][movieNumber] != 0:
            for key in testUserMovieRatings:
                if trainSet[i][key-1] == 0:
                    zeroCount+=1
            if zeroCount < count:
                newTrainSet.append(trainSet[i])

    #case where there are no common users
    if len(newTrainSet) == 0:
        print("the length of the training set:")
        print(len(newTrainSet))
        count = 0.0
        avg = 0.0
        for key in testUserMovieRatings:
            avg+=testUserMovieRatings[key]
            count+=1
        rating = float(avg/count)
        print(len(newTrainSet))
        print("rating was computed by doing the average due to no neighbors")
        return int(round(rating))

    for i in range(0,len(newTrainSet)):
        trainUserMovieRatings = {}

        #create new dictionary with movies test user rated and the ratings each training user gave those films
        for key in testUserMovieRatings:
            trainUserMovieRatings[key] = newTrainSet[i][key-1]

        #passing each dictionary into the cosine sim function
        similarity = cosineSimilarity(trainUserMovieRatings, testUserMovieRatings)
        preliminaryCosineValues.append(similarity)
        #givenRating = newTrainSet[i][movieNumber]

        #Discard values where cosine similarity is 1
        if similarity < 1:
            givenRating = newTrainSet[i][movieNumber]
            userRatingsDict[i] = givenRating
            userWeightsDict[i] = similarity

    #Computing weighted averages
    for key in userRatingsDict:
        numerator += userRatingsDict[key]*userWeightsDict[key]
        denominator += userWeightsDict[key]

    #case where there were no similar users to compare to
    if numerator == 0 or denominator == 0:
        count = 0.0
        avg = 0.0
        for key in testUserMovieRatings:
            avg+=testUserMovieRatings[key]
            count+=1.0
        rating = float(avg/count)
        print("rating was computed by doing the average")
    else:
        rating = float(numerator/denominator)
        print("rating was computed by cosine sim")

    return int(round(rating))



def runCosineSimilarity(trainSet, testSet, fileName):
    i = 0
    #userRatings = np.zeros(1000)
    movieRatingsDict = {}
    currentUser = testSet[0][0]
    fo = open(fileName,"w")

    while (i<len(testSet)):
        if testSet[i][0] != currentUser:
            currentUser = testSet[i][0]
            movieRatingsDict = {}

        if(testSet[i][2] == 0):
            rating = predictRating(testSet[i][1],trainSet,movieRatingsDict)
            #outputting to text file
            print(rating)
            fo.write(str(int(testSet[i][0])))
            fo.write(" ")
            fo.write(str(int(testSet[i][1])))
            fo.write(" ")
            fo.write(str(rating))
            fo.write("\n")
        else:
            #otherwise, we add the number to our test vector to be cosine simmed
            index = int(testSet[i][1])
            movieRatingsDict[index] = testSet[i][2]
        i+=1
    fo.close()
    return

# runCosineSimilarity(trainData,test5,"result5.txt")
# runCosineSimilarity(trainData,test10,"result10.txt")
# runCosineSimilarity(trainData,test20,"result20.txt")



################################################### PEARSON CORRELATION ##########################################################################################

def pearsonCorrelation(testUserMovieRatings, trainUserMovieRatings, testUserAvg, trainUserAvg):
    weight = 0.0
    numerator = 0.0
    d1 = 0.0
    d2 = 0.0
    denominator = 0.0
    count = 0.0

    for key in testUserMovieRatings:
        if testUserMovieRatings[key] != 0.0 and trainUserMovieRatings[key] != 0.0:
            numerator += (testUserMovieRatings[key]-testUserAvg)*(trainUserMovieRatings[key]-trainUserAvg)
            d1 += (testUserMovieRatings[key]-testUserAvg)*(testUserMovieRatings[key]-testUserAvg)
            d2 += (trainUserMovieRatings[key]-trainUserAvg)*(trainUserMovieRatings[key]-trainUserAvg)

    denominator = float(math.sqrt(d1))*float(math.sqrt(d2))

    if denominator != 0.0:
        return float(numerator/denominator)
    else:
        return 0.0



def predictPearsonCorrelationRating(movieNumber, trainSet, testUserMovieRatings):
    movieNumber -= 1
    newTrainSet = []
    count = len(testUserMovieRatings)
    testUserAvg = 0.0
    similarity = 0.0
    userRatingsDict = {}
    userWeightsDict = {}
    averageRatingsDict = {}
    similarityValues = []
    numerator = 0.0
    denominator = 0.0
    rating = 0.0
    count = 0.0
    trainUserMovieRatings = {}
    givenRatings = []
    ratingAverages = []

    for key in testUserMovieRatings:
        testUserAvg += testUserMovieRatings[key]
        count+=1
    testUserAvg /= count

    for i in range(0,len(trainSet)):
        zeroCount = 0
        #creating new array with training users that rated the movie we want to predict and rated the movies our test user rated
        if trainSet[i][movieNumber] != 0:
            for key in testUserMovieRatings:
                if trainSet[i][key-1] == 0:
                    zeroCount+=1
            if zeroCount < count:
                newTrainSet.append(trainSet[i])

    for i in range(0, len(newTrainSet)):
        trainUserMovieRatings = {}
        trainUserAvg = 0.0
        count2 = 0.0

        #create new dictionary with movies test user rated and the ratings each training user gave those films
        for key in testUserMovieRatings:
            trainUserMovieRatings[key] = newTrainSet[i][key-1]
            if trainUserMovieRatings[key] != 0:
                trainUserAvg+=trainUserMovieRatings[key]
                count2+=1

        trainUserAvg /= count2

        similarity = pearsonCorrelation(testUserMovieRatings, trainUserMovieRatings, testUserAvg, trainUserAvg)

        if similarity != 0.0:
            # similarityValues.append(similarity)
            givenRating = newTrainSet[i][movieNumber]
            userRatingsDict[i] = givenRating
            userWeightsDict[i] = similarity
            averageRatingsDict[i] = trainUserAvg

    for key in userRatingsDict:
        numerator += ((userRatingsDict[key]-averageRatingsDict[key])*abs((userWeightsDict[key])))
        denominator += abs(userWeightsDict[key])

    # print(numerator)
    # print(denominator)

    if numerator == 0 or denominator == 0:
        print("numerator or denominator was zero")
        rating = testUserAvg
        return int(round(rating))

    rating = testUserAvg + float(numerator/denominator)

    # if rating < 1.0 or rating > 5.0:
    #     ("numerator was under 1 or above 5")
    #     rating = float(testUserAvg)
    #     return int(round(rating))
    if rating < 1.0:
        return 1
    elif rating > 5.0:
        return 5

    return int(round(rating))



def runPearsonCorrelation(trainSet, testSet, fileName):
        i = 0
        movieRatingsDict = {}
        currentUser = testSet[0][0]

        fo = open(fileName,"w")

        while (i<len(testSet)):
            if testSet[i][0] != currentUser:
                currentUser = testSet[i][0]
                movieRatingsDict = {}

            if(testSet[i][2] == 0):
                rating = predictPearsonCorrelationRating(testSet[i][1], trainSet, movieRatingsDict)
                #outputting to text file
                print("rating:")
                print(rating)
                fo.write(str(int(testSet[i][0])))
                fo.write(" ")
                fo.write(str(int(testSet[i][1])))
                fo.write(" ")
                fo.write(str(rating))
                fo.write("\n")
            else:
                #otherwise, we add the number to our test vector to be cosine simmed
                index = int(testSet[i][1])
                #userRatings[index-1] = testSet[i][2]  #do index-1 because movies go from 1-1000 but array is zero indexed
                movieRatingsDict[index] = testSet[i][2]
            i+=1
        fo.close()
        return


# runPearsonCorrelation(trainData,test5,"result5.txt")
# runPearsonCorrelation(trainData,test10,"result10.txt")
# runPearsonCorrelation(trainData,test20,"result20.txt")



##################### PEARSON CORRELATION - INVERSE USER FREQUENCY MODIFICATION ############################################################

def calculateIUF(trainSet):
    iufValues = np.zeros(1000)

    for i in range(0, len(trainSet)):
        for j in range(0, len(trainSet[i])):
            if trainSet[i][j] > 0.0:
                iufValues[j] += 1

    for i in range(0, 1000):
        if iufValues[i] > 0:
            iufValues[i] = math.log10(200 / iufValues[i])

    return iufValues


def pearsonCorrelationIUF(testUserMovieRatings, trainUserMovieRatings, testUserAvg, trainUserAvg, IUF):
    weight = 0.0
    numerator = 0.0
    d1 = 0.0
    d2 = 0.0
    denominator = 0.0
    count = 0.0

    for key in testUserMovieRatings:
        if testUserMovieRatings[key] != 0.0 and trainUserMovieRatings[key] != 0.0:
            numerator += (IUF[key-1]*(testUserMovieRatings[key]-testUserAvg))*(IUF[key-1]*(trainUserMovieRatings[key]-trainUserAvg))
            d1 += (IUF[key-1]*(testUserMovieRatings[key]-testUserAvg))*(IUF[key-1]*(testUserMovieRatings[key]-testUserAvg))
            d2 += (IUF[key-1]*(trainUserMovieRatings[key]-trainUserAvg))*(IUF[key-1]*(trainUserMovieRatings[key]-trainUserAvg))

    denominator = float(math.sqrt(d1))*float(math.sqrt(d2))

    if denominator != 0.0:
        return numerator/denominator
    else:
        return 0.0



def predictPearsonCorrelationRatingIUF(movieNumber, trainSet, testUserMovieRatings, iufValues):
    movieNumber -= 1
    newTrainSet = []
    count = len(testUserMovieRatings) - 1
    testUserAvg = 0.0
    similarity = 0.0
    userRatingsDict = {}
    userWeightsDict = {}
    averageRatingsDict = {}
    similarityValues = []
    numerator = 0.0
    denominator = 0.0
    rating = 0.0
    count = 0.0
    trainUserMovieRatings = {}
    givenRatings = []
    ratingAverages = []

    for key in testUserMovieRatings:
        testUserAvg += testUserMovieRatings[key]
        count+=1
    testUserAvg /= count

    for i in range(0,len(trainSet)):
        zeroCount = 0
        #creating new array with training users that rated the movie we want to predict and rated the movies our test user rated
        if trainSet[i][movieNumber] != 0:
            for key in testUserMovieRatings:
                if trainSet[i][key-1] == 0:
                    zeroCount+=1
            if zeroCount < count:
                newTrainSet.append(trainSet[i])

    for i in range(0, len(newTrainSet)):
        trainUserMovieRatings = {}
        trainUserAvg = 0.0
        count2 = 0.0

        #create new dictionary with movies test user rated and the ratings each training user gave those films
        for key in testUserMovieRatings:
            trainUserMovieRatings[key] = newTrainSet[i][key-1]
            if trainUserMovieRatings[key] != 0:
                trainUserAvg+=trainUserMovieRatings[key]
                count2+=1

        trainUserAvg /= count2


        similarity = pearsonCorrelationIUF(testUserMovieRatings, trainUserMovieRatings, testUserAvg, trainUserAvg, iufValues)
        if similarity != 0.0:
            # similarityValues.append(similarity)
            givenRating = newTrainSet[i][movieNumber]
            userRatingsDict[i] = givenRating
            userWeightsDict[i] = similarity
            averageRatingsDict[i] = trainUserAvg

    for key in userRatingsDict:
        numerator += ((userRatingsDict[key]-averageRatingsDict[key])*abs(userWeightsDict[key]))
        denominator += abs(userWeightsDict[key])

    # print(numerator)
    # print(denominator)

    if numerator == 0 or denominator == 0:
        print(userWeightsDict)
        print(similarityValues)
        print("numerator or denominator was zero")
        rating = testUserAvg
        return int(round(rating))


    rating = testUserAvg + float(numerator/denominator)

    # if rating < 1.0 or rating > 5.0:
    #     ("numerator was under 1 or above 5")
    #     rating = float(testUserAvg)
    #     return int(round(rating))
    if rating < 1.0:
        return 1
    elif rating > 5.0:
        return 5

    # print(userWeightsDict)
    # print(userRatingsDict)
    # print(averageRatingsDict)
    return int(round(rating))



def runPearsonCorrelationIUF(trainSet, testSet, fileName):
        i = 0
        movieRatingsDict = {}
        currentUser = testSet[0][0]

        fo = open(fileName,"w")

        iufValues = calculateIUF(trainSet)

        while (i<len(testSet)):
            if testSet[i][0] != currentUser:
                currentUser = testSet[i][0]
                movieRatingsDict = {}

            if(testSet[i][2] == 0):
                rating = predictPearsonCorrelationRatingIUF(testSet[i][1], trainSet, movieRatingsDict, iufValues)
                #outputting to text file
                print("rating:")
                print(rating)
                fo.write(str(int(testSet[i][0])))
                fo.write(" ")
                fo.write(str(int(testSet[i][1])))
                fo.write(" ")
                fo.write(str(rating))
                fo.write("\n")
            else:
                #otherwise, we add the number to our test vector to be cosine simmed
                index = int(testSet[i][1])
                #userRatings[index-1] = testSet[i][2]  #do index-1 because movies go from 1-1000 but array is zero indexed
                movieRatingsDict[index] = testSet[i][2]
            i+=1
        fo.close()
        return


# runPearsonCorrelationIUF(trainData,test5,"result5.txt")
# runPearsonCorrelationIUF(trainData,test10,"result10.txt")
# runPearsonCorrelationIUF(trainData,test20,"result20.txt")



##################### PEARSON CORRELATION - CASE MODIFICATION ##########################################################################################

def predictPearsonCorrelationRatingCaseMod(movieNumber, trainSet, testUserMovieRatings):
    movieNumber -= 1
    newTrainSet = []
    count = len(testUserMovieRatings) - 1
    testUserAvg = 0.0
    similarity = 0.0
    userRatingsDict = {}
    userWeightsDict = {}
    averageRatingsDict = {}
    similarityValues = []
    numerator = 0.0
    denominator = 0.0
    rating = 0.0
    count = 0.0
    trainUserMovieRatings = {}
    givenRatings = []
    ratingAverages = []

    for key in testUserMovieRatings:
        testUserAvg += testUserMovieRatings[key]
        count+=1
    testUserAvg /= count

    for i in range(0,len(trainSet)):
        zeroCount = 0
        #creating new array with training users that rated the movie we want to predict and rated the movies our test user rated
        if trainSet[i][movieNumber] != 0:
            for key in testUserMovieRatings:
                if trainSet[i][key-1] == 0:
                    zeroCount+=1
            if zeroCount < count:
                newTrainSet.append(trainSet[i])

    for i in range(0, len(newTrainSet)):
        trainUserMovieRatings = {}
        trainUserAvg = 0.0
        count2 = 0.0

        #create new dictionary with movies test user rated and the ratings each training user gave those films
        for key in testUserMovieRatings:
            trainUserMovieRatings[key] = newTrainSet[i][key-1]
            if trainUserMovieRatings[key] != 0:
                trainUserAvg+=trainUserMovieRatings[key]
                count2+=1

        trainUserAvg /= count2


        similarity = pearsonCorrelation(testUserMovieRatings, trainUserMovieRatings, testUserAvg, trainUserAvg)
        if similarity != 0.0:
            similarityValues.append(similarity)
            #RUNNING CASE MOD -- can change constant
            similarity *= math.pow(abs(similarity),2.5)
            givenRating = newTrainSet[i][movieNumber]
            userRatingsDict[i] = givenRating
            userWeightsDict[i] = similarity
            averageRatingsDict[i] = trainUserAvg

    for key in userRatingsDict:
        numerator += ((userRatingsDict[key]-averageRatingsDict[key])*abs((userWeightsDict[key])))
        denominator += abs(userWeightsDict[key])

    # print(numerator)
    # print(denominator)

    if numerator == 0 or denominator == 0:
        print(userWeightsDict)
        print(similarityValues)
        print("numerator or denominator was zero")
        rating = testUserAvg
        return int(round(rating))


    rating = testUserAvg + float(numerator/denominator)

    if rating < 1.0 or rating > 5.0:
        ("numerator was under 1 or above 5")
        rating = float(testUserAvg)
        return int(round(rating))

    return int(round(rating))



def runPearsonCorrelationCaseMod(trainSet, testSet, fileName):
        i = 0
        movieRatingsDict = {}
        currentUser = testSet[0][0]

        fo = open(fileName,"w")

        while (i<len(testSet)):
            if testSet[i][0] != currentUser:
                currentUser = testSet[i][0]
                movieRatingsDict = {}

            if(testSet[i][2] == 0):
                rating = predictPearsonCorrelationRatingCaseMod(testSet[i][1], trainSet, movieRatingsDict)
                #outputting to text file
                print("rating:")
                print(rating)
                fo.write(str(int(testSet[i][0])))
                fo.write(" ")
                fo.write(str(int(testSet[i][1])))
                fo.write(" ")
                fo.write(str(rating))
                fo.write("\n")
            else:
                #otherwise, we add the number to our test vector to be cosine simmed
                index = int(testSet[i][1])
                #userRatings[index-1] = testSet[i][2]  #do index-1 because movies go from 1-1000 but array is zero indexed
                movieRatingsDict[index] = testSet[i][2]
            i+=1
        fo.close()
        return


# runPearsonCorrelationCaseMod(trainData,test5,"result5.txt")
# runPearsonCorrelationCaseMod(trainData,test10,"result10.txt")
# runPearsonCorrelationCaseMod(trainData,test20,"result20.txt")




##################### ITEM BASED FILTERING - ADJUSTED COSINE SIMILARITY ############################################################

def cosineSimilarityAdjusted(trainSet, testSet, avg):
    sumOfBoth = 0.0
    currSimilarity = 0.0
    sumOfTrain = 0.0
    sumOfTest = 0.0
    denominator = 0.0

    testUserRatings = []
    trainUserRatings = []

    for i in range(0,len(trainSet)):
        f = trainSet[i]
        b = testSet[i]

        if f!= 0 and b!=0:
            sumOfBoth += (f-avg)*(b-avg)
            sumOfTrain += math.pow(f-avg,2)
            sumOfTest += math.pow(b-avg,2)

    denominator = math.sqrt(sumOfTrain)*math.sqrt(sumOfTest)

    if denominator == 0:
        return 0
    else:
        return sumOfBoth/denominator

def predictRatingItemBased(movieNumber, trainSet, testSet, testUserMovieRatings):
    rating = 0.0
    preliminaryCosineValues = []
    userRatingsDict = {}
    userWeightsDict = {}
    numerator = 0.0
    denominator = 0.0
    newTrainSet = []
    movieNumber -= 1 #note: movies array is 1-1000 but array is 0 indexed
    count = len(testUserMovieRatings) - 1
    avg = 0.0

    trainSetTranposed = trainSet.transpose()
    testSetTransposed = testSet.transpose()

    for key in testUserMovieRatings:
        avg += testUserMovieRatings[key]
    avg /= float(len(testUserMovieRatings))

    for key in testUserMovieRatings:
        #passing in user ratings vector for each rating from test user as well as the ratings from the training set for the film we want a prediction for
        userWeightsDict[key] = cosineSimilarityAdjusted(trainSetTranposed[key-1], trainSetTranposed[movieNumber], avg)
        numerator += (userWeightsDict[key])*testUserMovieRatings[key]
        denominator += (userWeightsDict[key])

    if numerator == 0 or denominator == 0:
        rating = avg
        print("rating was computed by doing the average")
    else:
        rating = numerator/denominator
        print("rating was computed by cosine sim")

    if rating < 1.0 or rating > 5.0:
        rating = avg

    return int(round(rating))




def runCosineSimilarityItemBased(trainSet, testSet, fileName):
    i = 0
    userRatings = np.zeros(1000)
    movieRatingsDict = {}
    currentUser = testSet[0][0]

    # trainSetTranposed = trainSet.transpose()
    # testSetTransposed = testSet.transpose()

    fo = open(fileName,"w")

    while (i<len(testSet)):
        if testSet[i][0] != currentUser:
            currentUser = testSet[i][0]
            movieRatingsDict = {}

        if(testSet[i][2] == 0):
            rating = predictRatingItemBased(testSet[i][1], trainSet, testSet, movieRatingsDict)
            #outputting to text file
            print(rating)
            fo.write(str(int(testSet[i][0])))
            fo.write(" ")
            fo.write(str(int(testSet[i][1])))
            fo.write(" ")
            fo.write(str(rating))
            fo.write("\n")
        else:
            #otherwise, we add the number to our test vector to be cosine simmed
            index = int(testSet[i][1])
            userRatings[index-1] = testSet[i][2]  #do index-1 because movies go from 1-1000 but array is zero indexed
            movieRatingsDict[index] = testSet[i][2]
        i+=1
    fo.close()
    return


# runCosineSimilarityItemBased(trainData,test5,"result5.txt")
# runCosineSimilarity(trainData,test10,"result10.txt")
# runCosineSimilarity(trainData,test20,"result20.txt")



##################### MY OWN ALGORITHM ############################################################

def runSummationAlgo(trainSet, testSet, fileName):
    i = 0
    movieRatingsDict = {}
    currentUser = testSet[0][0]
    fo = open(fileName,"w")

    iufValues = calculateIUF(trainSet)

    while (i<len(testSet)):
        if testSet[i][0] != currentUser:
            currentUser = testSet[i][0]
            movieRatingsDict = {}

        if(testSet[i][2] == 0):
            rating1 = predictRating(testSet[i][1],trainSet,movieRatingsDict)
            rating2 = predictPearsonCorrelationRating(testSet[i][1], trainSet, movieRatingsDict)
            rating3 = predictPearsonCorrelationRatingIUF(testSet[i][1], trainSet, movieRatingsDict, iufValues)
            rating4 = predictPearsonCorrelationRatingCaseMod(testSet[i][1], trainSet, movieRatingsDict)
            rating5 = predictRatingItemBased(testSet[i][1], trainSet, testSet, movieRatingsDict)

            rating = int(round((rating1+rating2+rating3+rating5)/4))
            #outputting to text file
            print(rating)
            fo.write(str(int(testSet[i][0])))
            fo.write(" ")
            fo.write(str(int(testSet[i][1])))
            fo.write(" ")
            fo.write(str(rating))
            fo.write("\n")
        else:
            #otherwise, we add the number to our test vector to be cosine simmed
            index = int(testSet[i][1])
            movieRatingsDict[index] = testSet[i][2]
        i+=1
    fo.close()
    return



runSummationAlgo(trainData,test5,"result5.txt")
runSummationAlgo(trainData,test10,"result10.txt")
runSummationAlgo(trainData,test20,"result20.txt")







    #
