import os
import sentencepiece as spm
from  .base import Task

class SentencePiecingTask(Task):
    def __init__(self, config, language,  model_dir,  dataset_list, output_path):
        super(SentencePiecingTask, self).__init__(config, language)
        self.dataset_list = dataset_list
        self.output_path = output_path
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if output_path is not None and not os.path.exists(output_path):
            os.mkdir(output_path)


class SentencePiecingTrainTask(SentencePiecingTask):
    def __init__(self, config, language, model_dir, dataset_list):
        super(SentencePiecingTrainTask, self).__init__(
            config, language, model_dir, dataset_list, None)
        self.spm_trainer = spm.SentencePieceTrainer

    def _execute(self):

        input_ = ",".join(self.dataset_list)
        model_prefix = os.path.join(
            self.model_dir, self.language + "-spm-model")
        voc_size = self.config["params"]["vocab_size"]
        model_type = self.config["params"]["spm_model"]["model_type"]
        character_coverage = self.config["params"]["spm_model"]["character_coverage"][self.language]

        command = "--input=%s --model_prefix=%s --vocab_size=%d --character_coverage=%f --model_type=%s" % (
            input_, model_prefix, voc_size, character_coverage, model_type)

        self.spm_trainer.Train(command)


class SentencePiecingEncodingTask(SentencePiecingTask):
    def __init__(self, config, language, model_dir, dataset_list, output_path):
        super(SentencePiecingEncodingTask, self).__init__(
            config, language, model_dir, dataset_list, output_path)

    def _execute(self):
        assert (isinstance(self.dataset_list, list)
                ), "Please pass list as dataset list"
        model_path = os.path.join(
            self.model_dir, self.language + "-spm-model.model")
        spm_model = spm.SentencePieceProcessor()
        is_ok = spm_model.load(model_path)
        if not is_ok:
            raise ValueError("Unable to load model from %s" % model_path)
        for datafile in self.dataset_list:
            self.encode(spm_model, datafile)

    def encode(self, spm_model, data_file):
        output_format = self.config["params"]["spm_model"]["output_format"]
        if output_format == "id":
            encoding_method = spm_model.EncodeAsIds
        elif output_format == "piece":
            encoding_method = spm_model.EncodeAsPieces
        else:
            raise ValueError("Unsupported output format %s" % output_format)
        parent, data_file_name = os.path.split(data_file)
        with open(data_file) as d_file:
            data_lines = d_file.read().splitlines()

            with open(os.path.join(self.output_path, data_file_name), "w+") as output_file:
                for line in data_lines:
                    encoded = encoding_method(line)
                    encoded = " ".join(map(str, encoded))
                    output_file.write(encoded + "\n")
        return True


class SentencePiecingDecodingTask(SentencePiecingTask):
    def __init__(self, config, language, model_dir, dataset_list, output_path):
        super(SentencePiecingDecodingTask, self).__init__(
            config, language, model_dir, dataset_list, output_path)

    def _execute(self):
        assert (isinstance(self.dataset_list, list)
                ), "Please pass list as dataset list"
        model_path = os.path.join(
            self.model_dir, self.language + "-spm-model.model")
        spm_model = spm.SentencePieceProcessor()
        is_ok = spm_model.load(model_path)
        if not is_ok:
            raise ValueError("Unable to load model from %s" % model_path)
        for datafile in self.dataset_list:
            self.decode(spm_model, datafile)

    def decode(self, spm_model, data_file):
        output_format = self.config["params"]["spm_model"]["output_format"]
        if output_format == "id":
            decoding_method = spm_model.DecodeIds
        elif output_format == "piece":
            decoding_method = spm_model.DecodePieces
        else:
            raise ValueError("Unsupported output format %s" % output_format)

        parent, data_file_name = os.path.split(data_file)
        with open(data_file) as d_file:
            data_lines = d_file.read().splitlines()
            with open(os.path.join(self.output_path, data_file_name), "w+") as output_file:
                for line in data_lines:
                    if line == None or line.strip() == "":
                        output_file.write("\n")
                    else:
                        encoded = [int(i) for i in line.split(" ")]
                        decoded = decoding_method(encoded)
                        output_file.write(decoded + "\n")
        return True
