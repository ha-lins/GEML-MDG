#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/13 10:54
# @Author : kzl
# @Site :
# @File : poks.py

from torch.nn.modules.rnn import LSTMCell
from allennlp.nn.util import get_text_field_mask
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder,Seq2VecEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from CY_DataReadandMetric import *
from overrides import overrides
from allennlp.data.fields import Field, TextField, MetadataField, MultiLabelField, ListField
import torch
from allennlp.training.metrics import Average
import pkuseg

total_entiy = 78
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
            target_max_tokens: Optional[int] = 64,
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
        self.seg = pkuseg.pkuseg(model_name='/data3/linshuai/gen/cy', user_dict='/data3/linshuai/gen/cy/user_dict.txt')
        # self.seg = pkuseg.pkuseg(model_name='/data3/linshuai/gen/fd', user_dict='/data3/linshuai/gen/fd/idx2word.txt')
        # self.seg = pkuseg.pkuseg(model_name='medicine')

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
        response_tokens = response_tokens[-self._target_max_tokens:]
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
        fields["next_sym"] = MultiLabelField(list(sample['next_symp']), skip_indexing=True, num_labels=total_entiy)
        fields['target_tokens'] = TextField(response_tokens, self._target_token_indexers)
        fields['his_symptoms'] = MultiLabelField(list(sample['his_symp']), skip_indexing=True, num_labels=total_entiy)
        fields['tags'] = MetadataField(sample['tags'][-sen_num:])
        fields['history'] = ListField(fileds_list)
        # fields['dialog_index'] = MetadataField(sample['dialog_index'])

        return Instance(fields)


@Model.register("simple_seq2seq1")
class SimpleSeq2Seq(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            source_embedder: TextFieldEmbedder,
            encoder: Seq2VecEncoder,
            kg_encoder: Seq2VecEncoder,
            max_decoding_steps: int = 64,
            attention: Attention = None,
            target_namespace: str = "tokens",
            scheduled_sampling_ratio: float = 0.4,
    ) -> None:
        super().__init__(vocab)
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio  # Maybe we can try
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self.pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)
        self.hidden_dim = 300
        self._max_decoding_steps = max_decoding_steps
        self.kd_metric = KD_Metric()
        self.bleu_aver = NLTK_BLEU(ngram_weights=(0.25, 0.25, 0.25, 0.25))
        self.bleu1 = NLTK_BLEU(ngram_weights=(1, 0, 0, 0))
        self.bleu2 = NLTK_BLEU(ngram_weights=(0, 1, 0, 0))
        self.bleu4 = NLTK_BLEU(ngram_weights=(0, 0, 0, 1))
        self.topic_acc = Average()
        self.distinct1 = Distinct1()
        self.distinct2 = Distinct2()
        # anything about module
        self._source_embedder = source_embedder
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        target_embedding_dim = source_embedder.get_output_dim()
        self._target_embedder = Embedding(num_classes, target_embedding_dim)
        self._encoder = encoder
        self._kg_encoder = kg_encoder
        self._encoder_output_dim = self._encoder.get_output_dim()
        self._decoder_output_dim = self._encoder_output_dim
        # self.select_entity_num = 3
        self._decoder_input_dim = self.hidden_dim*2+total_entiy#self.select_entity_num
        self._attention = None
        if attention:
            self._attention = attention
            self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim

        self._decoder_cell = LSTMCell(self.hidden_dim * 2, self._decoder_output_dim)
        self._output_projection_layer = Linear(self.hidden_dim, num_classes)
        with open('cy/comp_topic2num.pk', 'rb') as f:
        # with open('fd/word2idx.pk', 'rb') as f:
            self.word_idx = pickle.load(f)
        self.vocab_to_idx = {}
        self.idx_to_vocab_list = []
        for word, k in self.word_idx.items():
            self.vocab_to_idx[vocab.get_token_index(word.strip())] = k
            self.idx_to_vocab_list.append(vocab.get_token_index(word.strip()))
        self.entity_size = total_entiy
        self.entity_embedding = torch.nn.Parameter(torch.Tensor(self.entity_size, self.hidden_dim))
        torch.nn.init.xavier_uniform_(self.entity_embedding, gain=1.414)
        self.entity_linear = Linear(self.hidden_dim*2, self.entity_size)
        self.gen_linear = Linear(self.hidden_dim, 1)


    @overrides
    def forward(self, tags, history, next_sym, source_tokens, his_symptoms, target_tokens, **args):
        bs = len(tags)
        embedded_input = self._source_embedder(source_tokens)
        source_mask = util.get_text_field_mask(source_tokens)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        kg_encoder_output = self._kg_encoder(embedded_input, source_mask)
        # if self.training == False:
        #     print(encoder_outputs[0])
        # final_encoder_output = util.get_final_encoder_states(encoder_outputs, source_mask, self._encoder.is_bidirectional())
        state = {
            "source_mask": source_mask,
            "encoder_outputs": encoder_outputs,
            "decoder_hidden": encoder_outputs,
            "decoder_context": encoder_outputs.new_zeros(bs, self._decoder_output_dim)
        }
        # find related entity
        # related_entity = torch.zeros(bs, self.entity_size).cuda()
        # his_symptom: bs * sym_size,   sym_mat: symp_size * symp_size
        # related_entity = his_symptoms.float().matmul(self.symp_mat)
        # # print(related_entity)
        # related_entity = (related_entity > 0.1).float()
        # # select_entity = related_entity.topk(self.select_entity_num)[1]
        # # entity_mask = torch.zeros_like(related_entity)
        # # for b in range(bs):
        # #     for i in range(self.select_entity_num):
        # #         entity_mask[b][select_entity[b][i]] = 1.
        # # print(select_entity)
        # stack_entity_embedding = self.entity_embedding.unsqueeze(0).repeat(bs,1, 1)  # bs * sym_size * hidden
        # context_hidden = kg_encoder_output.unsqueeze(1).repeat(1,self.entity_size, 1)  # bs * sym_size * hidden
        # entity_feature = torch.cat([stack_entity_embedding, context_hidden], dim=2)
        # print(entity_feature)
        # print(context_hidden.)
        entity_weight = torch.sigmoid(self.entity_linear(kg_encoder_output))  # bs * sym_size
        # print("entity_weight: ", entity_weight.size())
        topic_weight = torch.ones_like(next_sym) + 4 * next_sym
        # print("enity_weight: ",entity_weight.size())
        entity_loss = torch.nn.functional.binary_cross_entropy(entity_weight, next_sym.float(), weight=topic_weight.float())

        # if self.training:
        #     entity_weight = next_sym.float().unsqueeze(-1)
        # else:
        ans = (entity_weight > 0.5).long()
        entity_weight = ans.float()


        # entity_weight = entity_weight * entity_mask # bs, symp
        # norm_entity_weight = torch.zeros_like(entity_weight)
        # for b in range(bs):
        #     if entity_weight[b].sum() > 0.1:
        #         norm_entity_weight[b] = entity_weight[b] / entity_weight[b].sum()
        # if self.training == False:
        #     print(entity_weight[0])
        # entity_weight: bs   self.entity_embedding: sym_size * hidden
        # bs * sym_size * 1
        entity_hideen = entity_weight.squeeze(-1).matmul(self.entity_embedding)
        # entitity_hideen = entity_weight.unsqueeze(-1)
        # state["knowledge_hidden"] = torch.cat([entity_hideen, entity_weight.squeeze(-1)], dim=1)
        # state["knowledge_hidden"] = torch.cat([entity_hideen, entity_weight.squeeze(-1)], dim=1)
        state["knowledge_hidden"] = entity_hideen

        state["entity_prob"] = entity_weight
        # 获取一次decoder
        output_dict = self._forward_loop(state , target_tokens)
        best_predictions = output_dict["predictions"]

        # output something
        references, hypothesis = [], []
        for i in range(bs):
            cut_hypo = best_predictions[i][:]
            if self._end_index in list(best_predictions[i]):
                cut_hypo = best_predictions[i][:list(best_predictions[i]).index(self._end_index)]
            hypothesis.append([self.vocab.get_token_from_index(idx.item()) for idx in cut_hypo])
        flag = 1
        # for i in range(bs):
        #     cut_ref = target_tokens['tokens'][1:]
        #     if self._end_index in list(target_tokens['tokens'][i]):
        #         cut_ref = target_tokens['tokens'][i][1:list(target_tokens['tokens'][i]).index(self._end_index)]
        #     references.append([self.vocab.get_token_from_index(idx.item()) for idx in cut_ref])
        #     if random.random() <= 0.001 and not self.training and flag == 1:
        #         flag = 0
        #         for jj in range(i):
        #             print('___hypo___', ''.join(hypothesis[jj]), end=' ## ')
        #             print(''.join(references[jj]))
        #             print("")

        for i in range(bs):
            cut_ref = target_tokens['tokens'][1:]
            if self._end_index in list(target_tokens['tokens'][i]):
                cut_ref = target_tokens['tokens'][i][1:list(target_tokens['tokens'][i]).index(self._end_index)]
            references.append([self.vocab.get_token_from_index(idx.item()) for idx in cut_ref])
            if i == bs - 1 and not self.training and flag == 1:
                flag = 0
                history_mask = get_text_field_mask(history, num_wrapping_dims=1)
                utter_mask = get_text_field_mask(history)
                with open('save/human_eval/meta/poks_dis73_test1.txt', 'a+', encoding='utf-8') as f:
                    f.write("num of utter"+str(self.clac_num)+'\n')
                    for jj in range(bs):
                        for kx, aa in enumerate(utter_mask[jj]):
                            if aa != 0:
                                # print("sum: ",torch.sum(history_mask[jj][kx].long()).item())
                                # print("kx: ",kx)
                                # print(history[jj][kx])
                                # print("history: ",history[jj][kx][0].item())
                                # print(self.vocab.get_token_from_index(history['tokens'][jj][kx][0].item()))
                                bb = [self.vocab.get_token_from_index(history['tokens'][jj][kx][idx].item()) for idx in
                                      range(torch.sum(history_mask[jj][kx].long()).item())]
                                cc = ''.join(bb)
                                print(cc)
                                f.write(cc + '\n')
                        f.write("pre: " + ''.join(hypothesis[jj]) + "  ##GT:  " + ''.join(references[jj]) + '\n\n\n')
                        print('___hypo___', ''.join(hypothesis[jj]), end='  ##GT:  ')
                        print(''.join(references[jj]))
                        print("")



        self.bleu_aver(references, hypothesis)
        self.bleu1(references, hypothesis)
        self.bleu2(references, hypothesis)
        self.bleu4(references, hypothesis)
        self.kd_metric(references, hypothesis)#, dialog_index)
        self.distinct1(hypothesis)
        self.distinct2(hypothesis)
        if self.training:
            output_dict['loss'] += 12 * entity_loss
        return output_dict

    def _forward_loop(
            self, state: Dict[str, torch.Tensor], target_tokens: Dict[str, torch.LongTensor] = None
    ) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]
        batch_size = source_mask.size()[0]
        num_decoding_steps = self._max_decoding_steps
        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]
            _, target_sequence_length = targets.size()
            num_decoding_steps = target_sequence_length - 1

        if self.training:
            num_decoding_steps = target_sequence_length - 1

        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)  # (bs,)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                input_choices = last_predictions
            elif not target_tokens:
                input_choices = last_predictions
            else:
                input_choices = targets[:, timestep]
            #获取一次的decoder结果
            output_projections, state = self._prepare_output_projections(input_choices, state)  # bs * num_class
            step_logits.append(output_projections.unsqueeze(1))
            class_probabilities = F.softmax(output_projections, dim=-1)  # bs * num_class
            _, predicted_classes = torch.max(class_probabilities, 1)  # (bs,)

            last_predictions = predicted_classes
            step_predictions.append(last_predictions.unsqueeze(1))

        predictions = torch.cat(step_predictions, 1)  # bs * decoding_step

        output_dict = {"predictions": predictions}

        if self.training:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)
            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _prepare_output_projections(self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        encoder_outputs = state["encoder_outputs"]  # bs, seq_len, encoder_output_dim
        source_mask = state["source_mask"]  # bs * seq_len
        decoder_hidden = state["decoder_hidden"]  # bs, decoder_output_dim
        decoder_context = state["decoder_context"]  # bs * decoder_output

        embedded_input = self._target_embedder(last_predictions)  # bs * target_embedding
        # decoder_input = embedded_input

        decoder_input = torch.cat([embedded_input, state["knowledge_hidden"]], -1)


        # if self._attention:  # 如果加了seq_to_seq attention
        #     input_weights = self._attention(decoder_hidden, encoder_outputs, source_mask.float())  # bs * seq_len
        #     attended_input = util.weighted_sum(encoder_outputs, input_weights)  # bs * encoder_output
        #     decoder_input = torch.cat((attended_input, embedded_input), -1)  # bs * (decoder_output + target_embedding)
        # decoder_hidden = torch.cat([decoder_hidden, state["knowledge_hidden"]], -1)
        decoder_hidden, decoder_context = self._decoder_cell(
            decoder_input, (decoder_hidden, decoder_context)
        )
        state["decoder_hidden"] = decoder_hidden  # bs * hidden
        state["decoder_context"] = decoder_context


        # output_projections = self._output_projection_layer(torch.cat((decoder_hidden,graph_hidden),-1))
        output_projections = self._output_projection_layer(decoder_hidden)
        # output_projections_probs = torch.softmax(output_projections,1)
        # pgen = torch.sigmoid(self.gen_linear(decoder_hidden))
        # # if not self.training:
        # #     print('pgen : ', pgen)
        # output_projections_probs = pgen * output_projections_probs
        # for b in self.vocab_to_idx.keys():
        #     # print(state["entity_prob"][:, self.vocab_to_idx[b]])
        #     output_projections_probs[:, b] += (1 - pgen.squeeze(1)) * \
        #                                       state["entity_prob"][:, self.vocab_to_idx[b]].squeeze(1)
        # output_projections_probs[:, self.vocab.get_token_index("恶心")]=0
        # output_projections_probs[:, self.vocab.get_token_index("呕吐")]=0

        # history mask
        # for b in

        # if not self.training:
        #     print(output_projections_probs[:, self.vocab.get_token_index("恶心")])
        return output_projections, state

    @staticmethod
    def _get_loss(logits: torch.LongTensor, targets: torch.LongTensor, target_mask: torch.LongTensor) -> torch.Tensor:
        relevant_targets = targets[:, 1:].contiguous()
        relevant_mask = target_mask[:, 1:].contiguous()  # bs * decoding_step
        # return my_sequence_cross_entropy_with_logits(logits.contiguous(), relevant_targets, relevant_mask)
        return util.sequence_cross_entropy_with_logits(logits.contiguous(), relevant_targets, relevant_mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        all_metrics.update(self.kd_metric.get_metric(reset=reset))
        all_metrics.update({"BLEU_avg": self.bleu_aver.get_metric(reset=reset)})
        all_metrics.update({"BLEU1": self.bleu1.get_metric(reset=reset)})
        all_metrics.update({"d-1": self.distinct1.get_metric(reset=reset)})
        all_metrics.update({"d-2": self.distinct2.get_metric(reset=reset)})
        return all_metrics
