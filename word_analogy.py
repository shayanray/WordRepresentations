import os
import pickle
import numpy as np
import math
import sys


model_path = './models/'

#########################################################
##### Configurable Parameters for MODEL-TYPE and DEV/TEST file 
#loss_model = 'cross_entropy'
loss_model = 'nce'

#filename = 'word_analogy_dev.txt'
filename = 'word_analogy_test.txt'
#########################################################


model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb')) # change to 'r' if using 2.7, keep 'rb' for 3.x


# Remove the prediction results file if it exists already else ignore error
try:
    os.remove('predictions_'+loss_model+"_" + filename)
except OSError:
    pass


"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

def getMagnitude(vector):
    sum = 0.0
    for v in vector:
        sum += (v**2.0)

    return math.sqrt(sum)

def getCosineSimilarity(vector1, vector2):
    #print("vector 1 >> ", vector1)
    #print("vector 2 >> ", vector2)
    vector1_magnitude = getMagnitude(vector1) 
    vector2_magnitude = getMagnitude(vector2) 
    dotproduct = np.dot(vector1, vector2)
    cosine_similarity = dotproduct / (vector1_magnitude * vector2_magnitude)
    #print("cosine_similarity  ", cosine_similarity)
    return cosine_similarity 

def generatePredictions(filename):
    with open(filename) as filePtr:
        for aLine in filePtr.readlines():
            aLine = aLine.strip().replace("\"", "") #clear whitespaces and extra quotes
            examples = aLine.split("||")[0]
            choices = aLine.split("||")[1]
            ex_similarity = 0.0
            word_count = 0.0;

            #print("examples >> ", examples)
            #print("choices >> ", choices)

            # for each example pair, get the vector/embedding and compute the average similarity
            for aExPair in examples.split(","):
                exWords = aExPair.split(":")
                exWord1_vec = embeddings[dictionary[exWords[0]]]
                exWord2_vec = embeddings[dictionary[exWords[1]]]

                ex_similarity += getCosineSimilarity(exWord1_vec, exWord2_vec)
                word_count += 1.0

            avg_example_similarity = ex_similarity / word_count
            #print("avg_example_similarity  >> ", avg_example_similarity)

            choice_diffs = list()

            # for every choice pair, get the similarity and find its absolute difference between its value and average example similarity
            # get the minimum value (least illustrative) and maximum value (most illustrative) to get the results
            for aChoicePair in choices.split(","):
                choiceWords = aChoicePair.split(":")

                choiceWord1_vec = embeddings[dictionary[choiceWords[0]]]
                choiceWord2_vec = embeddings[dictionary[choiceWords[1]]]

                choice_diff = getCosineSimilarity(choiceWord1_vec, choiceWord2_vec)         
                choice_diff = np.abs(choice_diff - avg_example_similarity)

                choice_diffs.append(choice_diff)

            print("choice_diffs >> ", choice_diffs)
            minSimIndex = choice_diffs.index(min(choice_diffs))
            maxSimIndex = choice_diffs.index(max(choice_diffs))
            #print("minSimIndex >> "+ str(minSimIndex)+" maxSimIndex >> "+str(maxSimIndex))


            

            # write every line output to file
            with open('predictions_'+loss_model+"_" + filename, 'a') as predFile:
                for aChoicePair in choices.split(","):
                    predFile.write('"'+aChoicePair+'"' + ' ')

                # put the least then most
                predFile.write('"' +choices.split(",")[maxSimIndex] + '"' + ' ')
                predFile.write('"' +choices.split(",")[minSimIndex] + '"' + ' ')
                predFile.write("\n")

            print("Please check the output file >> ",'predictions_'+loss_model+"_" + filename)

if __name__ == "__main__":

    print("Usage :: python word_analogy.py")
    print("------------- Change filename and model-type in the code (1st few lines) -------------")

    # run predictions using given model
    generatePredictions(filename)
