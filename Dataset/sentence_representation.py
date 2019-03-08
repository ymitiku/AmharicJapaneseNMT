from .base import DataGenerator
import numpy as np

class SentenceDataGenerator(DataGenerator):
    def __init__(self, config, language):
        self.language = language
        super(SentenceDataGenerator, self).__init__(config)

    def _generate(self, case):

        if case == "train":
            return self.__generate_train()
        elif case == "dev":
            return self.__generate_dev()
        elif case == "test":
            return self.__generate_test()
        else:
            raise ValueError("Case value:'%s' is not recognized" % case)

    def __load_sentences(self, data_file):
        with open(data_file) as d_file:
            sentences = d_file.read().splitlines()
            sentences = [[int(i.strip()) for i in s if len(
                i.strip()) > 0] for s in sentences]
            return sentences

    def _load_dataset(self):
        train_file, dev_file, test_file = self._get_data_files()
        self.train_data = self.__load_sentences(train_file)
        self.dev_data = self.__load_sentences(dev_file)
        self.test_data = self.__load_sentences(test_file)

    def __generate_train(self):
        return self.__generate_data(self.train_data)

    def __generate_dev(self):
        return self.__generate_data(self.dev_data)

    def __generate_test(self):
        return self.__generate_data(self.test_data)

    def pad_sentence(self, sentence, max_length, end_of_sentence, pad_char):
        output = np.zeros((max_length, 1)) + pad_char
        if len(sentence) >= max_length:
            output[:max_length, :] = np.array(sentence[:max_length]).reshape(-1, 1)
            output[-1, 0] = end_of_sentence
        else:
            output[: len(sentence), :] = np.array(sentence).reshape(-1, 1)
            output[len(sentence), :] = end_of_sentence

        return output

    def __generate_data(self, dataset):
        max_length = self.config["params"]["sentence_representation"]["max_sentence_length"]
        end_of_sentence_index = self.config["params"]["sentence_representation"]["end_of_sentence_index"]
        pad_char_index = self.config["params"]["sentence_representation"]["pad_char_index"]
        batch_size = self.config["hyper_params"]["sentence_representation"]["batch_size"]
        while True:
            outputs = np.zeros(shape=(batch_size, max_length, 1))
            index = 0
            for s in dataset:
                padded = self.pad_sentence(
                    s, max_length, end_of_sentence_index, pad_char_index)
                outputs[index] = padded
                if index == batch_size-1:
                    o = outputs
                    index = 0
                    outputs = np.zeros(shape=(batch_size, max_length, 1))
                    yield o, o
                else:
                    index+=1
