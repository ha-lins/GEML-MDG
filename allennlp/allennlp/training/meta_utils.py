
from typing import Any, Union, Dict, Iterable, List, Optional, Tuple
import datetime
import json
import logging
import pathlib
import os
import shutil
from allennlp.data.vocabulary import Vocabulary
import torch
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel.scatter_gather import gather

from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.params import Params
from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.data.iterators.data_iterator import TensorDict
from allennlp.models.model import Model
from allennlp.models.archival import CONFIG_NAME
from allennlp.training.util import _set_up_cache_files
from allennlp.nn import util as nn_util

logger = logging.getLogger(__name__)

def meta_dataset_from_params(
    params: Params, cache_directory: str = None, cache_prefix: str = None
) -> Dict[str, Iterable[Instance]]:
    """
    Load all the datasets specified by the config.
    Parameters
    ----------
    params : ``Params``
    cache_directory : ``str``, optional
        If given, we will instruct the ``DatasetReaders`` that we construct to cache their
        instances in this location (or read their instances from caches in this location, if a
        suitable cache already exists).  This is essentially a `base` directory for the cache, as
        we will additionally add the ``cache_prefix`` to this directory, giving an actual cache
        location of ``cache_directory + cache_prefix``.
    cache_prefix : ``str``, optional
        This works in conjunction with the ``cache_directory``.  The idea is that the
        ``cache_directory`` contains caches for all different parameter settings, while the
        ``cache_prefix`` captures a specific set of parameters that led to a particular cache file.
        That is, if you change the tokenization settings inside your ``DatasetReader``, you don't
        want to read cached data that used the old settings.  In order to avoid this, we compute a
        hash of the parameters used to construct each ``DatasetReader`` and use that as a "prefix"
        to the cache files inside the base ``cache_directory``.  So, a given ``input_file`` would
        be cached essentially as ``cache_directory + cache_prefix + input_file``, where you specify
        a ``cache_directory``, the ``cache_prefix`` is based on the dataset reader parameters, and
        the ``input_file`` is whatever path you provided to ``DatasetReader.read()``.  In order to
        allow you to give recognizable names to these prefixes if you want them, you can manually
        specify the ``cache_prefix``.  Note that in some rare cases this can be dangerous, as we'll
        use the `same` prefix for both train and validation dataset readers.
    """
    dataset_reader_params = params.pop("dataset_reader")
    validation_dataset_reader_params = params.pop("validation_dataset_reader", None)
    train_cache_dir, validation_cache_dir = _set_up_cache_files(
        dataset_reader_params, validation_dataset_reader_params, cache_directory, cache_prefix
    )
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    train_data = []
    train_data_path = params.pop("train_data_path")
    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(
            validation_dataset_reader_params
        ) 
    for dataset in train_data_path:

        if train_cache_dir:
            dataset_reader.cache_data(train_cache_dir)
            validation_and_test_dataset_reader.cache_data(validation_cache_dir)
        logger.info("Reading training data from %s", train_data_path)
        train_data.append(dataset_reader.read(dataset))

      
    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_data_path = params.pop("validation_data_path", None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets["test"] = test_data
    return datasets

    def make_vocab(serialization_dir :str, recover :bool, all_datasets):
        if recover and os.path.exists(os.path.join(serialization_dir, "vocabulary")):
            vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
            params.pop("vocabulary", {})
        else:
            vocab = Vocabulary.from_params(
                params.pop("vocabulary", {}),
                # Using a generator comprehension here is important
                # because, being lazy, it allows us to not iterate over the
                # dataset when directory_path is specified.
                (
                    instance
                    for key, dataset in all_datasets.items()
                    if key in datasets_for_vocab_creation
                    for instance in dataset
                ),
            )
        return vocab