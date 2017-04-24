import json
from collections import defaultdict
import sys
import datetime



class HMMDecoder:
    
    def __init__(self):
        self.transitionProbabilities = None
        self.emissionProbabilities = None
        self.predefinedTags = None
    
    def readModel(self, modelFile):
        with open(modelFile) as model_output:
            modelObj = json.load(model_output)
        self.emissionProbabilities = modelObj['Emission Probabilities']
        self.transitionProbabilities = modelObj['Transition Probabilities']
        self.predefinedTags = modelObj['PredefinedTags']

    def decode(self,testFile, modelFile, outputFile):
        self.readModel(modelFile)
        fptr = open(outputFile, 'w')
       
        wordTransitionEmissiondict = defaultdict(int)
        for line in open(testFile,"r",encoding="utf8"):
            tagSequence = []
            tagSeq = ""
            prevTagList = {}  
            words  = line.strip().split(' ')
            if words[0] not in self.emissionProbabilities.keys():
                for possibleWordTag in self.predefinedTags:
                    wordTransitionEmissiondict[possibleWordTag] = self.transitionProbabilities[possibleWordTag]['START']
                    prevTagList[(possibleWordTag,0)] =  'START'
            else:
                for possibleWordTag in self.emissionProbabilities[words[0]].keys():
                        wordTransitionEmissiondict[possibleWordTag] = self.emissionProbabilities[words[0]][possibleWordTag] + self.transitionProbabilities[possibleWordTag]['START']
                        prevTagList[(possibleWordTag,0)] =  'START'
                       
             
            for itr in range(1,len(words)):
                tempWorddict  = {}
                emission_dict= {}
                if words[itr] not in self.emissionProbabilities.keys():
                    for possibleWordTag in self.predefinedTags:
                        emission_dict[possibleWordTag] = 0.0
                    self.emissionProbabilities[words[itr]] = emission_dict
                        
                for currentTag in self.emissionProbabilities[words[itr]].keys():
                    maximumProbability = float('-Inf')
                    for previousTag in wordTransitionEmissiondict.keys():
                        probability = wordTransitionEmissiondict[previousTag] + self.transitionProbabilities[currentTag][previousTag]
                        if probability > maximumProbability:
                            maximumProbability = probability
                            mostLikelyTag = previousTag
                    tempWorddict[currentTag] = maximumProbability + self.emissionProbabilities[words[itr]][currentTag]
                    prevTagList[(currentTag,itr)] = mostLikelyTag
                wordTransitionEmissiondict = tempWorddict
                tagSequence.append(prevTagList)

            mostLikelyTag = max(wordTransitionEmissiondict, key=wordTransitionEmissiondict.get)
            tagSeq = words[itr] + "/" + mostLikelyTag + " " + tagSeq
            for i in range(len(words)-1,0,-1):
                prevTag = prevTagList[(mostLikelyTag,i)]
                tagSeq =  words[i-1] + "/"+ prevTag + " " +tagSeq 
                mostLikelyTag = prevTag
            fptr.write(tagSeq)
            fptr.write("\n")
        fptr.close()

if __name__ == "__main__":
    hmmDecoder = HMMDecoder()
    now = datetime.datetime.now()
    print("Decoding Started",str(now))
    testFile = sys.argv[1]  #"./hw5-data-corpus/catalan_corpus_dev_raw.txt" 
    modelFile = "hmmmodel.txt"
    outputFile = "hmmoutput.txt"
    hmmDecoder.decode(testFile, modelFile, outputFile)
    now = datetime.datetime.now()
    print("Decoded Ended",str(now))
    