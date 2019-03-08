from Tasks import SentencePiecingTrainTask, SentencePiecingEncodingTask,\
    SentencePiecingDecodingTask, Word2VecTrainingTask, \
    RWord2VecTrainingTask, S2STrainingTask
from .config import load_config
import os

def __create_encode_decode_task(args, config):
    model_dir = config["output"]["spm"]["model_dir"]

    if args.task == "spm_encode":
        output_path = config["output"]["spm"]["encoded_dir"]
        task = SentencePiecingEncodingTask(
            config, args.language, model_dir, dataset_list, output_path)
    else:
        output_path = config["output"]["spm"]["decoded_dir"]
        encoded_path = config["output"]["spm"]["encoded_dir"]
        d_list = os.listdir(encoded_path)
        dataset_list = []
        for l in d_list:
            if l.endswith("." + args.language):
                dataset_list.append(os.path.join(encoded_path, l))
        task = SentencePiecingDecodingTask(
            config, args.language, model_dir, dataset_list, output_path)

    return task
def __create_training_task(args, config):
    if args.train_task == "word2vec":
        return Word2VecTrainingTask(config, args.language)
    elif args.train_task == "rword2vec":
        return RWord2VecTrainingTask(config, args.language)
    elif args.train_task == "sentence":
        return S2STrainingTask(config, args.language)
    else:
        raise NotImplementedError("Not implemented for %s training task"%args.train_task)
def create_task(args):
    config = load_config(args.config)
    dataset_list = [config["dataset"]["raw_data"][args.language]
                    ["train"], config["dataset"]["raw_data"][args.language]["dev"], config["dataset"]["raw_data"][args.language]["test"]]
    if args.task == "spm_train":
        model_dir = config["output"]["spm"]["model_dir"]
        spm_train_task = SentencePiecingTrainTask(config, args.language, model_dir, dataset_list)
        return spm_train_task
    elif args.task == "spm_encode" or args.task == "spm_decode":
        return __create_encode_decode_task(args, config)
    
    elif args.task == "train":
        return __create_training_task(args, config) 
    else:
        raise NotImplementedError("Task %s is not implemented"%args.task)


    
