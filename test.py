import json
import sys

def read_data(dataset_path, version):
    with open(dataset_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != version):
            print('Evaluation expects v-' + version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    return dataset

def open_dataset(dataset):
    f = open('./data/questions.txt', 'w')
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                f.write(qa['question'] + '\n')
    f.close()


dataset_path = './data/train-v1.1.json'
version = '1.1'

dataset = read_data(dataset_path, version)
open_dataset(dataset)
