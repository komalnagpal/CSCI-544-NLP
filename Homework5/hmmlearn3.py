from collections import defaultdict
import collections
import json,math
import sys
import datetime

class HMMLearner:
    def __init__(self,inputFile):
        self.inputFile = inputFile
        self.tagCountGivenTag = defaultdict(lambda : defaultdict(int))
        self.tagCount = defaultdict(int)
        self.transitionProbabilities = defaultdict(lambda: defaultdict(float))
        self.wordGiventagCount = defaultdict(lambda: defaultdict(int))
        self.emissionProbabilities = defaultdict(lambda: defaultdict(float))
        self.predefinedTags =set()

    
    def readInputFile(self):
        for line in open(self.inputFile,"r",encoding="utf8"):
            taggedWords = line.strip().split(' ')
            self.tagCountGivenTag['START'][taggedWords[0][-2:]] += 1
            for itr in range(len(taggedWords) - 1):
                current_tag = taggedWords[itr][-2:]
                self.predefinedTags.add(current_tag)
                next_tag = taggedWords[itr+1][-2:]
                self.tagCountGivenTag[current_tag][next_tag] +=1
                self.tagCount[current_tag] += 1
                self.wordGiventagCount[taggedWords[itr][:-3]][current_tag] += 1
            self.predefinedTags.add(taggedWords[itr + 1][-2:])
            self.tagCount[taggedWords[itr + 1][-2:]] += 1
            self.wordGiventagCount[taggedWords[itr + 1][:-3]][taggedWords[itr+1][-2:]] += 1

            
    def addOneSmoothing(self):
        for currentTag, nextTagCount in self.tagCountGivenTag.items():
            for tag in self.predefinedTags:
                if tag not in nextTagCount.keys():
                    self.tagCountGivenTag[currentTag][tag] = 1.0
                    
    def calculateTransitionProbabilites(self):
        for currentTag, nextTagCount in self.tagCountGivenTag.items():
            for tag in self.predefinedTags:
                if tag not in nextTagCount.keys():
                    self.tagCountGivenTag[currentTag][tag] = 1.0
            for tag,count in nextTagCount.items():
                self.transitionProbabilities[tag][currentTag] =  math.log(float(count) / sum(nextTagCount.values()))
    
    def calculateEmissionProbabilities(self):
        for word, tagCount in self.wordGiventagCount.items():
            for tag,count in tagCount.items():
                self.emissionProbabilities[word][tag] =math.log(float(count) / self.tagCount[tag],10)


    def saveModel(self,outputFile):
        model_params = collections.OrderedDict()
        model_params['Transition Probabilities'] = self.transitionProbabilities
        model_params['Emission Probabilities'] = self.emissionProbabilities
        model_params['PredefinedTags'] = list(self.predefinedTags)
        with open(outputFile,"w") as fp:
            json.dump(model_params,fp,indent = 4)
            
if __name__ == "__main__":
    
    inputFile = sys.argv[1]
    now = datetime.datetime.now()
    print("Modelling Started",str(now))
    hmmModel = HMMLearner(inputFile)
    hmmModel.readInputFile()
    hmmModel.calculateTransitionProbabilites()
    hmmModel.calculateEmissionProbabilities()
    hmmModel.saveModel("hmmmodel.txt")
    now = datetime.datetime.now()
    print("Modelling Ended",str(now))