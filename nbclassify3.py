import sys
import utilities as util
import json
import math


class NaiveBayesClassifier:
    def __init__(self, stop_words, symbols):
        self.stop_words = stop_words
        self.symbols = symbols
        self.prior_probabilities = None
        self.likekihood = None
        self.model_dict ={}

    def read_model(self, model_file):
        with open(model_file) as model_output:
            model_params = json.load(model_output)
            self.model_dict = model_params["modelinfo"]
            self.prior_probabilities = model_params["prior"]
            self.likekihood = model_params["likelihood"]

    def predict(self, test_file, model_file, output_file):
        self.read_model(model_file)
        fptr = open(output_file, 'w')
        with open(test_file) as text:
            for line in text:
                line_id, line_text = line.strip().split(' ', 1)
                words_in_text = util.tokenize(line_text, self.stop_words)
                result = ""
                for model_name, class_labels in self.model_dict.items():
                    predicted_label = self.predict_class_label(words_in_text, class_labels)
                    result = result + " " + predicted_label
                result = str(line_id) + result
                fptr.write(result + "\n")
        fptr.close()


    def predict_class_label(self,words_in_text,class_labels):
        posterior_probabilities = {}
        for class_label in class_labels:
            posterior_probabilities[class_label] = math.log(
                                                   self.prior_probabilities[
                                                       class_label]) + self.predict_text_label(
                words_in_text, class_label)
        return max(posterior_probabilities,
                   key=posterior_probabilities.get)

    def predict_text_label(self, tokens, class_label):
        text_likelihood = 0.0
        for token in tokens:
            if token in self.likekihood:
                if class_label in self.likekihood[token]:
                    text_likelihood += math.log(
                        self.likekihood[token][class_label])
                else:
                    continue
            else:
                continue
        return text_likelihood


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 1:
        print("Test Data File Not Found")
        sys.exit()
    print("Check")
    nb_classifier = NaiveBayesClassifier([], [])
    test_file = sys.argv[1]
    model_file = "nbmodel.txt"
    output_file = "nboutput.txt"
    nb_classifier.predict(test_file, model_file, output_file)
