import argparse
from Utils import create_task

task_choices = [
    "spm_train",
    "spm_encode",
    "spm_decode",
    "train"
]
training_task_choices = [
    "word2vec",
    "sentence",
    "translation"
]


def main(args):
    task = create_task(args)
    task.execute()
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', choices=task_choices, default='train', required=True)
    parser.add_argument('-k', '--train_task', choices=training_task_choices)
    parser.add_argument('-c', '--config', default='config/config.yml')
    parser.add_argument('-l', '--language', default='am')

    args = parser.parse_args()
    main(args)