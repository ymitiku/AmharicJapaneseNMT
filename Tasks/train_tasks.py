from .base import Task
from Networks import Word2VecModel, ReverseWord2VecModel, S2SModel
from Dataset import Word2VecDataGenerator, ReverseWord2VecDataGenerator, SentenceDataGenerator


training_models = {
    "word2vec": Word2VecModel,
    "rword2vec": ReverseWord2VecModel,
    "sentence": S2SModel
}
training_data_generators = {
    "word2vec": Word2VecDataGenerator,
    "rword2vec": ReverseWord2VecDataGenerator,
    "sentence": SentenceDataGenerator
}

class TrainingTask(Task):
    def __init__(self, config, language, model_type):
        super(TrainingTask, self).__init__(config, language)
        self.model = self._get_model(model_type)(config, language)
        self.data_generator = self._get_generator(model_type)(config, language)
    def _get_model(self, model_type):
        return training_models[model_type]
    def _get_generator(self, model_type):
        return training_data_generators[model_type]
    
class Word2VecTrainingTask(TrainingTask):
    def __init__(self, config, language):
        super(Word2VecTrainingTask, self).__init__(config, language, "word2vec")
    def _execute(self):
        self.model.train(self.data_generator.generate(), None, True)
class RWord2VecTrainingTask(TrainingTask):
    def __init__(self, config, language):
        super(RWord2VecTrainingTask, self).__init__(
            config, language, "rword2vec")
    def _execute(self):
        self.model.train(self.data_generator.generate(), None, True)


class S2STrainingTask(TrainingTask):
    def __init__(self, config, language):
        super(S2STrainingTask, self).__init__(
            config, language, "sentence")

    def _execute(self):
        self.model.train(self.data_generator.generate(
            case="train"), self.data_generator.generate(case="dev"), True)
