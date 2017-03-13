import sys
import json


def read_data(dataset_path, version):
    with open(dataset_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != version):
            print('Evaluation expects v-' + version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    return dataset


def train(model, params, dataset):
    print('### Training ###')
    print("\nTraining loss: %.3f, f1: %.3f, em: %.3f, ep: %d" % 
            (0, 0, 0, 0))



def test(model, params, dataset):
    print('### Testing ###')
    print("\nTesting loss: %.3f, f1: %.3f, em: %.3f, ep: %d" % 
            (0, 0, 0, 0))

    
'''
for article in dataset:
    for idx, paragraph in enumerate(article['paragraphs']):
        context = paragraph['context']
        qas = paragraph['qas']
'''

