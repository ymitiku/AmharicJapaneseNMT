from .base import Task
from Networks import Word2VecModel
from Dataset import Word2VecDataGenerator


training_models = {
    "word2vec":Word2VecModel
}
training_data_generators = {
    "word2vec": Word2VecDataGenerator
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
        self.model.train(self.data_generator.generate(), None,True)