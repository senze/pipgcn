import numpy as np
from config import info

# export modules
__all__ = ['Fitter']


class Fitter:
    """ Fitter of models """
    def __init__(self, processor=None):
        self.processor = processor

    def fit(self, exp_spec, data, model):
        """ trains model by iterating mini-batches for specified number of epochs """
        # train for specified number of epochs
        for epoch in range(1, exp_spec['num_epochs'] + 1):
            info('epoch_' + str(epoch))
            self.train_epoch(data['train'], model, exp_spec['mini_batch_size'])
        # calculate train and test metrics
        headers, result = self.processor.process_results(exp_spec, data, model, 'epoch_' + str(exp_spec['num_epochs']))
        # clean up
        self.processor.reset()
        model.close()
        return headers, result

    def train_epoch(self, data, model, mini_batch_size):
        """
        Trains model for one pass through training data, one protein at a time
        Each protein is split into mini-batches of paired examples.
        Features for the entire protein is passed to model, but only a mini-batch of examples are passed
        """
        random_data_indices = np.random.permutation(len(data))
        # loop through each protein
        for protein_index in random_data_indices:
            # extract just data for this protein
            protein_data = data[protein_index]
            pair_examples = protein_data['label']
            n = len(pair_examples)
            random_pair_indices = np.random.permutation(np.arange(n))
            # loop through each mini-batch
            for i in range(int(n / mini_batch_size)):
                # extract data for this mini-batch
                index = int(i * mini_batch_size)
                examples = pair_examples[random_pair_indices[index: index + mini_batch_size]]
                mini_batch_data = {}
                for feature_type in protein_data:
                    if feature_type == 'label':
                        mini_batch_data['label'] = examples
                    else:
                        mini_batch_data[feature_type] = protein_data[feature_type]
                # train the model
                model.train(mini_batch_data)
