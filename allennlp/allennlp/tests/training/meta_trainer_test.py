# pylint: disable=invalid-name,too-many-public-methods,protected-access
import copy
import glob
import json
import os
import re
import time
from typing import Dict
import torch
import pytest
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, ModelTestCase
from allennlp.training import MetaTrainer
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.util import sparse_clip_norm
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.data.iterators import BasicIterator
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader#, WikiTablesDatasetReader
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.training.moving_average import ExponentialMovingAverage


class TestTrainer(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        # TODO make this a set of dataset readers
        # Classification may be easier in this case. Same dataset reader but with different paths 
        self.instances_list = []
        self.instances_list.append(SequenceTaggingDatasetReader().read(self.FIXTURES_ROOT / 'data' / 'meta_seq' / 'sequence_tagging.tsv'))
        self.instances_list.append(SequenceTaggingDatasetReader().read(self.FIXTURES_ROOT / 'data' / 'meta_seq' / 'sequence_tagging1.tsv'))
        self.instances_list.append(SequenceTaggingDatasetReader().read(self.FIXTURES_ROOT / 'data' / 'meta_seq' / 'sequence_tagging2.tsv'))
        # loop through dataset readers and extend vocab
        combined_vocab = Vocabulary.from_instances(self.instances_list[0])

        for instance in self.instances_list:
            combined_vocab.extend_from_instances(Params({}), instances=instance)
        self.vocab = combined_vocab
        # Figure out params TODO 
        self.model_params = Params({
                "text_field_embedder": {
                        "token_embedders": {
                                "tokens": {
                                        "type": "embedding",
                                        "embedding_dim": 5
                                        }
                                }
                        },
                "encoder": {
                        "type": "lstm",
                        "input_size": 5,
                        "hidden_size": 7,
                        "num_layers": 2
                        }
                })
        self.model = SimpleTagger.from_params(vocab=self.vocab, params=self.model_params)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.01, momentum=0.9)
        self.iterator = BasicIterator(batch_size=2)
        self.iterator.index_with(combined_vocab)

    def test_trainer_can_run(self):
        trainer = MetaTrainer(model=self.model,
                          optimizer=self.optimizer,
                          iterator=self.iterator,
                          train_datasets=self.instances_list,
                          validation_datasets=self.instances_list[0],
                          num_epochs=2)
        metrics = trainer.train()
        assert 'best_validation_loss' in metrics
        assert isinstance(metrics['best_validation_loss'], float)
        assert 'best_validation_accuracy' in metrics
        assert isinstance(metrics['best_validation_accuracy'], float)
        assert 'best_validation_accuracy3' in metrics
        assert isinstance(metrics['best_validation_accuracy3'], float)
        assert 'best_epoch' in metrics
        assert isinstance(metrics['best_epoch'], int)

        # Making sure that both increasing and decreasing validation metrics work.
        trainer = MetaTrainer(model=self.model,
                          optimizer=self.optimizer,
                          iterator=self.iterator,
                          train_datasets=self.instances_list,
                          validation_datasets=self.instances_list,
                          validation_metric='+loss',
                          num_epochs=2, 
                          meta_batches=3)
        metrics = trainer.train()
        assert 'best_validation_loss' in metrics
        assert isinstance(metrics['best_validation_loss'], float)
        assert 'best_validation_accuracy' in metrics
        assert isinstance(metrics['best_validation_accuracy'], float)
        assert 'best_validation_accuracy3' in metrics
        assert isinstance(metrics['best_validation_accuracy3'], float)
        assert 'best_epoch' in metrics
        assert isinstance(metrics['best_epoch'], int)
        assert 'peak_cpu_memory_MB' in metrics
        assert isinstance(metrics['peak_cpu_memory_MB'], float)
        assert metrics['peak_cpu_memory_MB'] > 0

