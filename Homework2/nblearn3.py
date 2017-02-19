import sys
import math
import string
import collections
import json
import utilities as util

class NaiveBayesLearner:
    def __init__(self, stop_words, symbols):
        self.stop_words = stop_words
        self.symbols = symbols
        self.prior_probabilities = {}
        self.likelihood = collections.defaultdict(dict)

    def read_input_data(self, input_text_file, input_label_file):
        text_dict = {}
        class_count_dict = {}
        total_samples = 0
        word_class_count_dict = collections.defaultdict(
            lambda: collections.defaultdict(int))
        class_words_count_dict = collections.defaultdict(int)
        with open(input_text_file) as text:
            for line in text:
                line_id, line_text = line.strip().split(' ', 1)
                text_dict[line_id] = util.tokenize(line_text, self.stop_words)
        with open(input_label_file) as labels:
            for line in labels:
                total_samples += 1
                line_id, label_1, label_2 = line.strip().split()
                if label_1 in class_count_dict:
                    class_count_dict[label_1] += 1
                else:
                    class_count_dict[label_1] = 1
                if label_2 in class_count_dict:
                    class_count_dict[label_2] += 1
                else:
                    class_count_dict[label_2] = 1
                if label_1 + " " + label_2 in class_count_dict:
                    class_count_dict[label_1 + " " + label_2] += 1
                else:
                    class_count_dict[label_1 + " " + label_2] = 1
                for word in text_dict[line_id]:
                    word_class_count_dict[word][label_1] += 1
                    word_class_count_dict[word][label_2] += 1
                    word_class_count_dict[word][
                        label_1 + " " + label_2] += 1
                    class_words_count_dict[label_1] += 1
                    class_words_count_dict[label_2] += 1
                    class_words_count_dict[label_1 + " " + label_2] += 1
        for k,v in word_class_count_dict.items():
            print(k,v,"\n")
        return class_count_dict, word_class_count_dict, class_words_count_dict

    def tokenize(self, text):
        text = text.replace('\n', '')
        translator = str.maketrans('', '', string.punctuation)
        tokens = text.strip().translate(translator).lower().strip().split()
        preprocessed_tokens = [token.strip() for token in tokens if
                               token not in self.stop_words]
        return preprocessed_tokens

    def calculate_prior_probabilities(self,class_count, class_labels):
        total_samples = sum([class_count[label] for label in class_labels if label in class_count])
        for label in class_labels:
            if label in class_count:
                self.prior_probabilities[label] = float(
                    class_count[label])/total_samples

    def learn(self, train_file, label_file, class_label_dict):
        class_count, word_class_count, class_words_count_dict = self.read_input_data(
            train_file, label_file)
        for model_name, class_labels in class_label_dict.items():
            self.calculate_prior_probabilities(class_count, class_labels)
            self.calculate_conditional_probability(word_class_count, class_labels, class_words_count_dict)

    def calculate_conditional_probability(self, word_class_count,class_labels, class_words_count_dict):
        for word, class_count in word_class_count.items():
            for label in class_labels:
                # print(word, label, class_count[label],len(word_class_count.keys()),class_words_count_dict[label])
                result = float(class_count[label] + 1.0) / (
                    len(word_class_count.keys()) + class_words_count_dict[
                        label])
                self.likelihood[word][label] = result

    def save_model(self, output_file,class_label_dict ):
        model_params = collections.OrderedDict()
        model_params["modelinfo"] = class_label_dict
        model_params["prior"]= self.prior_probabilities
        model_params["likelihood"] = self.likelihood
        with open(output_file, 'w') as outfile:
            json.dump(model_params, outfile, indent=8)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Input Parameters missing")
    train_text_file = sys.argv[1]
    train_label_file = sys.argv[2]
    nb_learner = NaiveBayesLearner([], [])
    model_dict = collections.OrderedDict()
    model_dict["mod1"] =["positive","negative"]
    model_dict["mod2"] = ["truthful" ,"deceptive"]
    # model_dict = {"single Model": ["yes T","no F","yes F","no T"]}
    nb_learner.learn(train_text_file, train_label_file,model_dict)
    nb_learner.save_model("nbmodel.txt",model_dict)
