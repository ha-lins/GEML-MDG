from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField ,ListField
from allennlp.data.instance import Instance
import pickle
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Tuple, Dict, Optional
from collections import Counter
import math
from typing import Iterable, Tuple, Dict, Set,List
import pkuseg
from allennlp.data.fields import Field, TextField, MetadataField, MultiLabelField, ListField
from overrides import overrides
import torch
from allennlp.training.metrics.metric import Metric
import random
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
total_entiy = 77
@DatasetReader.register("seqreader")
class Seq2SeqDatasetReader(DatasetReader):

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_add_start_token: bool = True,
        delimiter: str = "\t",
        source_max_tokens: Optional[int] = 256,
        target_max_tokens: Optional[int] = 32,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.pre_sen = 10
        self.seg = pkuseg.pkuseg(model_name='medicine', user_dict='../data/0510/mdg/user_dict.txt')
        # self.max_tokens = 150

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            for sample in dataset:
                yield self.text_to_instance(sample)

    @overrides
    def text_to_instance(self, sample) -> Instance:
        fields: Dict[str, Field] = {}
        sen_num = self.pre_sen
        context = ' '.join(sample['history'][-sen_num:])
        all_sentence = sample['history'][-sen_num:]
        # history = ' '.join(list(''.join(context)))
        history = ' '.join(self.seg.cut(context))

        text_tokens = self._source_tokenizer.tokenize(history)
        text_tokens = text_tokens[-self._source_max_tokens:]
        text_tokens.insert(0, Token(START_SYMBOL))
        text_tokens.append(Token(END_SYMBOL))

        # response = ' '.join(sample['response'])
        response = ' '.join(self.seg.cut(sample['response']))
        response_tokens = self._target_tokenizer.tokenize(response)
        response_tokens = response_tokens[:self._target_max_tokens]
        response_tokens.insert(0, Token(START_SYMBOL))
        response_tokens.append(Token(END_SYMBOL))

        fileds_list = []
        for sen in all_sentence:
            sen = ' '.join(self.seg.cut(sen))
            # sen = ' '.join(sen)
            txt_token = self._source_tokenizer.tokenize(sen)
            ff = TextField(txt_token,self._source_token_indexers)
            fileds_list.append(ff)
        fields['source_tokens'] = TextField(text_tokens, self._source_token_indexers)
        fields["next_sym"] = MultiLabelField(list(sample['next_sym']), skip_indexing=True, num_labels=77)
        fields['target_tokens'] = TextField(response_tokens, self._target_token_indexers)
        fields['his_symptoms'] = MultiLabelField(list(sample['history_tag']), skip_indexing=True, num_labels=77)
        # fields['future_sym'] = MultiLabelField(list(sample['future_sym']), skip_indexing=True, num_labels=77)
        fields['tags'] = MetadataField(sample['tags'][-sen_num:])
        fields['history'] = ListField(fileds_list)
        fields['dialog_index'] = MetadataField(sample['dialog_index'])
        fields['future_sym'] = MetadataField(sample['future_sym'])
        return Instance(fields)



@Metric.register("knowledge")
class KD_Metric(Metric):
    def __init__(self) -> None:
        self._pred_true = 0
        self._total_pred = 0
        self._total_true = 0
        self.future_pred_true = 0.
        self.pred_diease = {}
        self.true_diease = {}
        with open('../data/0510/mdg/biii_sym_dict.pk', 'rb') as f:
            self.norm_dict = pickle.load(f)

    def reset(self) -> None:
        self._pred_true = 0
        self._total_pred = 0
        self._total_true = 0
        self.pred_diease = {}
        self.true_diease = {}
        self.future_pred_true = 0.

    @overrides
    def get_metric(self, reset: bool = False):
        rec, acc, f1 = 0., 0., 0.
        drec, dacc, df1 = 0., 0., 0.
        facc = 0.
        # print("pred_true",self._pred_true)
        # print("_total_pred",self._total_pred)
        # print("_total_true",self._total_true)
        if self._total_pred > 0:
            acc = self._pred_true / self._total_pred
            facc = self.future_pred_true / self._total_pred
        if self._total_true > 0:
            rec = self._pred_true / self._total_true
        if acc > 0 and rec > 0:
            f1 = acc * rec * 2 / (acc + rec)

        p_t, t_t = len(self.pred_diease), len(self.true_diease)
        t_p = 0
        for k,v in self.pred_diease.items():
            if self.true_diease.get(k,-1)==v:
                t_p += 1
        # print("pred: ",p_t)
        # print("true: ",t_t)
        # print("pred_true: ",t_p)

        if p_t > 0:
            dacc = t_p / p_t
        if t_t > 0:
            drec = t_p / t_t
        if dacc > 0 and drec > 0:
            df1 = dacc * drec * 2 / (dacc + drec)

        if reset:
            self.reset()
        return {'rec': rec, 'acc': acc, 'f1': f1, 'drec': drec, 'dacc': dacc, 'df1': df1, "fcc": facc}

    def convert_sen_to_entity_set(self, sen):
        entity_set = set()
        for entity in self.norm_dict.keys():
            if entity in sen:
                entity_set.add(self.norm_dict[entity])
        return entity_set

    @overrides
    def __call__(
        self,
        references, # list(list(str))
        hypothesis, # list(list(str))
        dialog_index,  # list(int)
        future_sym,
    ) -> None:
        # print("len: ",len(references))
        for batch_num in range(len(references)):
            ref = ''.join(references[batch_num])
            hypo = ''.join(hypothesis[batch_num])
            ref_list = self.convert_sen_to_entity_set(ref)
            hypo_list = self.convert_sen_to_entity_set(hypo)
            # print("pred_true", self._pred_true)
            # print("_total_pred", self._total_pred)
            # print("_total_true", self._total_true)
            # print("ref: ",len(ref_list))
            # print("hypo: ",len(hypo_list))
            d_r, d_h = [], []
            for r in hypo_list:
                if r < 6:
                    self.pred_diease[dialog_index[batch_num]] = r
            for h in ref_list:
                if h < 6:
                    self.true_diease[dialog_index[batch_num]] = h

            self._total_true += len(ref_list)
            self._total_pred += len(hypo_list)

            for entity in hypo_list:
                if entity in ref_list:
                    self._pred_true += 1
                if entity in future_sym[batch_num]:
                    self.future_pred_true += 1

@Metric.register("nltk_bleu")
class NLTK_BLEU(Metric):
    def __init__(
        self,
        ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
    ) -> None:
        self._ngram_weights = ngram_weights
        self._scores = []
        self.smoothfunc = SmoothingFunction().method7
        # if all(ngram_weights = SmoothingFunction().method0

    def reset(self) -> None:
        self._scores = []

    @overrides
    def get_metric(self, reset: bool = False):
        score = 0.
        if len(self._scores):
            score = sum(self._scores) / len(self._scores)
        if reset:
            self.reset()
        return score

    @overrides
    def __call__(
        self,
        references, # list(list(str))
        hypothesis, # list(list(str))
    ) -> None:
        for batch_num in range(len(references)):
            if len(hypothesis[batch_num]) <= 4:
                self._scores.append(0)
            else:
                self._scores.append(sentence_bleu([references[batch_num]], hypothesis[batch_num],
                                          smoothing_function=self.smoothfunc,
                                          weights=self._ngram_weights))

def my_sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       weights: torch.FloatTensor,
                                       average: str = "batch",
                                       label_smoothing: float = None,
                                      ) -> torch.FloatTensor:
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")

    # make sure weights are float
    weights = weights.float()
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.log(logits_flat + 1e-16)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()
    # focal loss coefficient

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        num_non_empty_sequences = ((weights_batch_sum > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights_batch_sum.sum() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        return per_batch_loss


@Metric.register("my_average")
class MyAverage(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """

    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    @overrides
    def __call__(self, value, num):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        self._total_value += list(self.unwrap_to_tensors(value))[0]
        self._count += num

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0


@Metric.register("my_F1")
class F1(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """

    def __init__(self) -> None:
        self._total_value = 0.0

    @overrides
    def __call__(self, value):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        self._total_value = list(self.unwrap_to_tensors(value))[0]

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = self._total_value
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0

@Metric.register("distinct1")
class Distinct1(Metric):
    def __init__(self):
        self._total_vocabs = 0
        self.appear_vocabs = set()
    @overrides
    def __call__(self, hypothesis):
        batch_size = len(hypothesis)
        for b in range(batch_size):
            self._total_vocabs += len(hypothesis[b])
            self.appear_vocabs.update(hypothesis[b])

    def reset(self) -> None:
        print("-------------------------------")
        print("-"*100)
        # print(self._total_vocabs)
        # print(self.appear_vocabs)
        self._total_vocabs = 0
        self.appear_vocabs = set()

    def get_metric(self, reset: bool = False):
        value = len(self.appear_vocabs) / self._total_vocabs
        if reset:
            self.reset()
        return value

@Metric.register("distinct2")
class Distinct2(Metric):
    def __init__(self):
        self._total_vocabs = 0
        self.appear_vocabs = set()

    @overrides
    def __call__(self, hypothesis):
        batch_size = len(hypothesis)
        for b in range(batch_size):
            if len(hypothesis[b]) <= 1:
                continue
            self._total_vocabs += len(hypothesis[b]) - 1
            for i in range(len(hypothesis[b])-1):
                self.appear_vocabs.add(hypothesis[b][i]+hypothesis[b][i+1])

    def reset(self) -> None:
        # print("-------------------------------")
        # print("-"*1000)
        # print(self._total_vocabs)
        # print(self.appear_vocabs)

        self._total_vocabs = 0
        self.appear_vocabs = set()

    def get_metric(self, reset: bool = False):
        value = len(self.appear_vocabs) / self._total_vocabs
        if reset:
            self.reset()
        return value