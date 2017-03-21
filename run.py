import tensorflow as tf


def train(model, params, dataset):
    print('### Training ###')
    
    batch_size = params['batch_size']
    sess = model.session
    for batch_idx in range(0, len(dataset), batch_size):
        batch_data = dataset[batch_idx:batch_idx+batch_size]
        batch_context = batch_data['context']
        batch_question = batch_data['question']
        batch_answer = batch_data['answer']
    
    print("\nTraining loss: %.3f, f1: %.3f, em: %.3f, ep: %d" % 
            (0, 0, 0, 0))



def test(model, params, dataset):
    print('### Testing ###')
    print("\nTesting loss: %.3f, f1: %.3f, em: %.3f, ep: %d" % 
            (0, 0, 0, 0))

