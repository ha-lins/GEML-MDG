from torch.nn.modules.rnn import LSTMCell
from allennlp.nn.util import get_text_field_mask
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder,Seq2VecEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from CY_DataReadandMetric import *
from overrides import overrides
from allennlp.data.fields import Field, TextField, MetadataField, MultiLabelField, ListField
import torch
from allennlp.training.metrics import Average
import pkuseg
import warnings
warnings.filterwarnings("ignore")

# 句子变成10句



total_entity = 78
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
        # self.seg = pkuseg.pkuseg(model_name='medicine', user_dict='../data/0510/cy/user_dict.txt')
        # self.max_tokens = 150
        self.seg = pkuseg.pkuseg(model_name='/data3/linshuai/gen/cy', user_dict='/data3/linshuai/gen/cy/user_dict.txt')

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
        fields["next_sym"] = MultiLabelField(list(sample['next_symp']), skip_indexing=True, num_labels=total_entity+sen_num)
        fields['target_tokens'] = TextField(response_tokens, self._target_token_indexers)
        fields['his_symptoms'] = MultiLabelField(list(sample['his_symp']), skip_indexing=True, num_labels=total_entity+sen_num)
        fields['tags'] = MetadataField(sample['tags'][-sen_num:])
        fields['history'] = ListField(fileds_list)
        return Instance(fields)


class GATAttention(torch.nn.Module):
    def __init__(self, in_feature, out_feature, n_head):
        super(GATAttention, self).__init__()
        self.infeature = in_feature
        self.outfeature = out_feature
        self.n_head = n_head
        self.extend_w = torch.nn.Linear(self.infeature,self.outfeature,bias=False)
        self.A = torch.nn.Linear(self.outfeature*2, 1, bias=False)
        self.leakyrelu = torch.nn.LeakyReLU()

    def forward(self, node_embedding, att_mat):
        # symp_hidden : bs, symp_size, att_mat
        # att_mat : bs, symp_size, symp_size
        att_mat = att_mat.cuda()
        symp_hidden = self.extend_w(node_embedding)
        bs, symp_size = symp_hidden.size()[0], symp_hidden.size()[1]
        alpha_input = torch.cat([symp_hidden.repeat(1, 1, symp_size).view(bs, symp_size * symp_size, -1),
                                 symp_hidden.repeat(1, symp_size, 1)], dim=2)
        alpha_input = alpha_input.view(bs, symp_size, symp_size, 2 * self.outfeature)
        alpha = self.A(alpha_input).squeeze(-1) # bs, symp, symp
        alpha = torch.tanh(alpha)

        diag = torch.eye(symp_size).cuda()
        diag = diag.unsqueeze(0).repeat(bs, 1, 1)
        invinf = torch.zeros_like(att_mat) - 1e15
        # alpha = alpha.masked_fill(diag == 1, -1e8)
        attention = torch.where(att_mat > 0, alpha, invinf)
        # attention = attention.masked_fill(diag == 1, -1e10)
        # aadv = (adv > 0).float()
        # attention = alpha * att_mask
        attention = torch.nn.functional.softmax(attention, dim=-1)

        last_h = attention.matmul(symp_hidden)  # bs * sym_sz * embeed
        su = torch.sum(att_mat, dim=1).unsqueeze(-1)
        # print("su: ", su.size())
        # print("node_embedding: ", node_embedding.size())
        # print("last_h: ", last_h.size())
        last_h = torch.where(su==0, node_embedding, last_h)
        # for i in range(bs):
        #     for j in range(symp_size):
        #         if torch.sum(att_mat[i,:,j]).item()==0:
        #             last_h[i][j] = node_embedding[i][j]
        return last_h


@Model.register("simple_seq2seq1")
class SimpleSeq2Seq(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            source_embedder: TextFieldEmbedder,
            encoder: Seq2SeqEncoder,
            vecoder: Seq2VecEncoder,
            sen_encoder: Seq2VecEncoder,
            max_decoding_steps: int = 32,
            attention: Attention = None,
            beam_size: int = None,
            target_namespace: str = "tokens",
            scheduled_sampling_ratio: float = 0.5,
    ) -> None:
        super().__init__(vocab)
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio  # Maybe we can try
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self.pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)
        self._max_decoding_steps = max_decoding_steps
        self.vocab = vocab
        # anything about dims
        self.sen_num = 10
        # with open('../data/0510/cy/kg_and_train.pk', 'rb') as f:
        with open('cy/openkg.pk', 'rb') as f:
            self.kg_mat = torch.tensor(pickle.load(f)).float()
        self.symp_mat = torch.nn.Parameter(self.kg_mat).cuda()
        self.evovl_mat = torch.zeros(len(self.kg_mat), len(self.kg_mat)).cuda()

        # with open('../data/0510/cy/comp_topic2num.pk', 'rb') as f:
        with open('cy/comp_topic2num.pk', 'rb') as f:
            self.word_idx = pickle.load(f)
        self.idx_word = {v:k for k, v in self.word_idx.items()}
        self.vocab_to_idx = {}
        self.idx_to_vocab_list = []
        self.vocab_list = []
        for word, k in self.word_idx.items():
            self.vocab_to_idx[vocab.get_token_index(word.strip())] = k
            self.idx_to_vocab_list.append(vocab.get_token_index(word.strip()))

        self.symp_size = len(self.symp_mat) + self.sen_num
        self.topic = len(self.symp_mat)
        self._encoder = encoder
        self._vecoder = vecoder
        self._sen_encoder = sen_encoder

        self.outfeature = self._sen_encoder.get_output_dim()
        # anything about graph
        self.symp_state = torch.nn.Parameter(torch.Tensor(self.symp_size, self.outfeature))
        torch.nn.init.xavier_uniform_(self.symp_state, gain=1.414)
        self.predict_layer = torch.nn.Parameter(torch.Tensor(self.symp_size, self.outfeature))
        self.predict_bias = torch.nn.Parameter(torch.Tensor(self.symp_size))
        torch.nn.init.kaiming_uniform_(self.predict_layer)
        torch.nn.init.uniform_(self.predict_bias, -1 / self.symp_size ** 0.5, 1 / self.symp_size ** 0.5)

        self.attn_one = GATAttention(self.outfeature, self.outfeature, 1)
        self.attn_two = GATAttention(self.outfeature, self.outfeature, 1)
        self.attn_three = GATAttention(self.outfeature, self.outfeature, 1)

        # Metric
        self.kd_metric = KD_Metric()
        self.bleu_aver = NLTK_BLEU(ngram_weights=(0.25, 0.25, 0.25, 0.25))
        self.bleu1 = NLTK_BLEU(ngram_weights=(1, 0, 0, 0))
        self.bleu2 = NLTK_BLEU(ngram_weights=(0, 1, 0, 0))
        self.bleu4 = NLTK_BLEU(ngram_weights=(0, 0, 0, 1))
        self.topic_acc = Average()
        # anything about module
        self._source_embedder = source_embedder
        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        target_embedding_dim = source_embedder.get_output_dim()
        self._target_embedder = Embedding(num_classes, target_embedding_dim)
        self._encoder_output_dim = self._encoder.get_output_dim() # 600  要不把前两个都换成outfeater得了
        self._decoder_output_dim = self._encoder_output_dim * 2
        self._decoder_input_dim = target_embedding_dim
        self._attention = None
        if attention:
            self._attention = attention
            self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim

        # 在这里把那个embedding融合进入试试？
        self.before_linear = Linear(2 * self.outfeature,self.outfeature)
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        self._output_projection_layer = Linear(self.outfeature * 2, num_classes)

        self.linear_all = Linear(self.outfeature * 3 + self._decoder_input_dim,1)
        self.attention_linear = Linear(self.outfeature, self.outfeature)
        self.decoder_linear = Linear(self.outfeature * 2, self.outfeature)

        self.get_attn = Linear(self.outfeature, 1,bias=False)
        self.topic_acc = MyAverage()
        self.topic_rec = MyAverage()
        self.topic_f1 = F1()
        self.dink1 = Distinct1()
        self.dink2 = Distinct2()
        self.last_sen = 2

    @overrides
    def forward(self, tags, history, next_sym, source_tokens, his_symptoms, target_tokens, **args):
        bs = len(tags)
        # self.flatten_parameters()
        # 获取history的embedding
        embeddings = self._source_embedder(history)
        mask = get_text_field_mask(history, num_wrapping_dims=1)  # num_wrapping 增加维度
        sz = list(embeddings.size())
        embeddings = embeddings.view(sz[0] * sz[1], sz[2], sz[3])
        mask = mask.view(sz[0] * sz[1], sz[2])

        # 获取每一句的hidden  bs * sen_num * hidden
        utter_hidden = self._vecoder(embeddings, mask)
        utter_hidden = utter_hidden.view(sz[0], sz[1], -1)  # bs * sen_num * hidden

        dialog_mask = get_text_field_mask(history)
        dialog_hidden = self._sen_encoder(utter_hidden, dialog_mask)  # hred的形式
        # print("dialog_hidden: ",dialog_hidden.size())

        # 初始化每个节点
        symp_state = torch.zeros(bs, self.symp_size, self.outfeature).cuda()  # bs * symp_size * hidden
        symp_state += self.symp_state   # 每一个节点的初始化emb相同，这是个问题吗？

        # 句子与句子连边  如果不用cuda呢
        sym_mat = torch.zeros(bs, self.symp_size, self.symp_size)

        for i in range(bs):
            dic = {}
            for j in range(len(tags[i])):  # 这一个地方可以改一下，这里是和前面的所有有关系的边都连上了
                symp_state[i][self.topic + j] = utter_hidden[i][j]
                dic[j] = set(list(tags[i][j]))
                for k in range(j):
                    for aa in dic[j]:
                        if aa in dic[k] and aa != -1:
                            # sym_mat[i][self.topic+j][self.topic+k] += 1
                            sym_mat[i][self.topic+k][self.topic+j] += 1

        for i in range(bs):
            for j in range(len(tags[i])):
                symp_state[i][self.topic + j] = utter_hidden[i][j]
                last = min(j+self.last_sen, len(tags[i]))
                sym_mat[i][j][j:last] = torch.ones(1)   # 和后面几条边相连

        last_h = self.attn_one(symp_state, sym_mat)
        sym_mat = torch.zeros(bs, self.symp_size, self.symp_size)
        for i in range(bs):
            for j in range(len(tags[i])):
                for tt in tags[i][j]:
                    if tt != -1:
                        sym_mat[i][self.topic + j][tt] += 1
        #
        last_h = self.attn_two(last_h, sym_mat)
        #
        # topic和topic连边
        sym_mat = torch.zeros(bs, self.symp_size, self.symp_size)

        for symp_i in his_symptoms:
            for symp_j in his_symptoms:
                self.evovl_mat[symp_i][symp_j] = 1
        temp_mat = (torch.nn.functional.relu(self.symp_mat) + self.evovl_mat).cpu()
        with open('visulize_graph.txt', 'a') as fout:
            fout.write('evovl_mat is: \n')
            for i in self.evovl_mat.detach().cpu().numpy():
                fout.write(str(i) + '\n')
            fout.write('temp_mat is: \n')
            for i in temp_mat.detach().cpu().numpy():
                fout.write(str(i) + '\n')
        # print('[info] temp_mat is:{}'.format(temp_mat))
        sym_mat[:, :self.topic, :self.topic] += temp_mat

        # sym_mat[:, :self.topic, :self.topic] += self.symp_mat
        # last_h = self.attn_two(symp_state, sym_mat)
        last_h = self.attn_three(last_h, sym_mat)

        topic_pre = torch.sum(self.predict_layer * last_h, dim=-1) + self.predict_bias
        topic_probs = torch.sigmoid(topic_pre)
        topics_weight = torch.ones_like(topic_probs) + 5 * next_sym.float()
        topic_loss = torch.nn.functional.binary_cross_entropy(topic_probs, next_sym.float(), weight=topics_weight)

        ans = (topic_probs > 0.5).long()

        # his_symptoms bs * sym_size?
        # his_mask = torch.where(his_symptoms > 0, torch.full_like(his_symptoms, 0), torch.full_like(his_symptoms,1)).long()

        # 隐藏句子节点
        # his_mask
        his_sentence_mask = torch.zeros(bs, self.sen_num).long()
        total_mask = torch.cat((torch.ones(bs, self.topic).long(), his_sentence_mask), -1)

        if self.training:
            aa = next_sym.long()
        else:
            aa = ans
        # total_mask = torch.ones(bs, self.symp_size).cuda()
        # total_mask = total_mask.long() & his_mask.long()
        topic_embedding = aa.float().matmul(self.symp_state)
        topic_hidden = last_h

        # 计算topic的f1, acc, rec
        pre_total = torch.sum(ans).item()
        true_total = torch.sum(next_sym).item()
        pre_right = torch.sum((ans == next_sym).long() * next_sym).item()
        # print(pre_total,pre_right)
        self.topic_acc(pre_right, pre_total)
        self.topic_rec(pre_right, true_total)
        acc = self.topic_acc.get_metric(False)
        rec = self.topic_rec.get_metric(False)
        f1 = 0.
        if acc + rec > 0:
            f1 = acc * rec * 2 / (acc + rec)
        self.topic_f1(f1)

        # Encoding source_tokens
        # embedded_input = self._source_embedder(source_tokens)
        source_mask = util.get_text_field_mask(source_tokens)
        # encoder_outputs = self._encoder(embedded_input, source_mask)
        # final_encoder_output = util.get_final_encoder_states(encoder_outputs, source_mask, self._encoder.is_bidirectional())
        state = {
            "source_mask": source_mask,
            "encoder_outputs": topic_embedding,  # bs * seq_len * dim
            "decoder_hidden":  torch.cat((topic_embedding, dialog_hidden), -1),  # bs * dim hred的输出
            # "decoder_hidden": torch.sum(last_h * total_mask.float(), 1),
            "decoder_context": topic_embedding.new_zeros(bs, self._decoder_output_dim),
            # "decoder_context": topic_embedding,
            "topic_embedding": topic_embedding
        }
        # state[''] = topic_embedding
        # 获取一次decoder
        output_dict = self._forward_loop(state, topic_hidden, total_mask.cuda(), target_tokens)
        best_predictions = output_dict["predictions"]

        # output something
        references, hypothesis = [], []
        for i in range(bs):
            cut_hypo = best_predictions[i][:]
            if self._end_index in list(best_predictions[i]):
                cut_hypo = best_predictions[i][:list(best_predictions[i]).index(self._end_index)]
            hypothesis.append([self.vocab.get_token_from_index(idx.item()) for idx in cut_hypo])

        flag = 1
        for i in range(bs):
            cut_ref = target_tokens['tokens'][1:]
            if self._end_index in list(target_tokens['tokens'][i]):
                cut_ref = target_tokens['tokens'][i][1:list(target_tokens['tokens'][i]).index(self._end_index)]
            references.append([self.vocab.get_token_from_index(idx.item()) for idx in cut_ref])
            if random.random() <= 0.001 and not self.training and flag == 1:
                flag = 0
                for jj in range(i):
                    print('___hypo___', ''.join(hypothesis[jj]), end=' ## ')
                    print(''.join(references[jj]))
                    print("")

        self.bleu_aver(references, hypothesis)
        self.bleu1(references, hypothesis)
        self.bleu2(references, hypothesis)
        self.bleu4(references, hypothesis)
        self.kd_metric(references, hypothesis)
        self.dink1(hypothesis)
        self.dink2(hypothesis)
        if self.training:
            output_dict['loss'] = output_dict['loss'] + 8 * topic_loss
        else:
            output_dict['loss'] = topic_loss
        return output_dict

    def _forward_loop(
            self, state: Dict[str, torch.Tensor], topic: torch.Tensor, total_mask: torch.Tensor, target_tokens: Dict[str, torch.LongTensor] = None
    ) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]
        num_decoding_steps = self._max_decoding_steps
        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]
            _, target_sequence_length = targets.size()
            if self.training:
                num_decoding_steps = target_sequence_length - 1

        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)  # (bs,)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                input_choices = targets[:, timestep]
            else:
                input_choices = last_predictions
            #获取一次的decoder结果
            output_projections, state = self._prepare_output_projections(input_choices, state, topic, total_mask)  # bs * num_class
            step_logits.append(output_projections.unsqueeze(1))
            # class_probabilities = F.softmax(output_projections, dim=-1)  # bs * num_class
            class_probabilities = output_projections
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

    def _prepare_output_projections(self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor],
                                    topic: torch.Tensor, total_mask: torch.Tensor):

        encoder_outputs = state["encoder_outputs"]  # bs, seq_len, encoder_output_dim
        source_mask = state["source_mask"]  # bs * seq_len
        decoder_hidden = state["decoder_hidden"]  # bs, decoder_output_dim
        decoder_context = state["decoder_context"]  # bs * decoder_output

        embedded_input = self._target_embedder(last_predictions)  # bs * target_embedding
        decoder_input = embedded_input
        if self._attention:  # 如果加了seq_to_seq attention
            input_weights = self._attention(decoder_hidden, encoder_outputs, source_mask.float())  # bs * seq_len
            attended_input = util.weighted_sum(encoder_outputs, input_weights)  # bs * encoder_output
            decoder_input = torch.cat((attended_input, embedded_input), -1)  # bs * (decoder_output + target_embedding)

        # 换一种attention的方式
        # topic_new = topic.view(-1,self.outfeature)
        sz = list(topic.size())   # bs * sym_size * hidden
        decoder_hideen_new = decoder_hidden.unsqueeze(1)
        decoder_hideen_new = decoder_hideen_new.repeat(1, sz[1], 1)
        decoder_hideen_new = decoder_hideen_new.view(sz[0] * sz[1], -1)
        topic_hideen_new = topic.view(-1, sz[2])
        # total_attention_need = torch.cat((decoder_hideen_new, topic_hideen_new), -1)
        # after_linear = self.attention_linear(total_attention_need)
        after_linear = self.decoder_linear(decoder_hideen_new) + self.attention_linear(topic_hideen_new)
        logi = self.get_attn(torch.tanh(after_linear))
        logi = logi.view(-1,sz[1])
        # print("logi: ",logi[0:2])

        # 应该在这一个地方加入his_mask
        # print("total_masK: ",total_mask[0])
        logi = logi.masked_fill(total_mask == 0, -1e9)
        probs = F.softmax(logi, dim=-1)  # bs * sym_size  #在这里还是mask了这张图，让多余的节点消失了
        probs = probs.unsqueeze(-1)

        graph_attention = probs * topic  # bs * sym_size * hideen
        graph_hidden = torch.sum(graph_attention, 1)  # h*

        # decoder_input = torch.cat((decoder_input, state['topic_embedding']), -1)
        decoder_hidden, decoder_context = self._decoder_cell(
            decoder_input, (decoder_hidden, decoder_context)
        )  # 可以尝试直接融入到decoder_intput里面

        state["decoder_hidden"] = decoder_hidden  # bs * hidden
        state["decoder_context"] = decoder_context

        # one_linear_out = self.before_linear(torch.cat((decoder_hidden, graph_hidden),-1))
        # output_projections = self._output_projection_layer(one_linear_out)  # P(w)  bs * num_class
        output_projections = self._output_projection_layer(decoder_hidden)
        output_projections_probs = F.softmax(output_projections, dim=-1)

        all_hidden = torch.cat((graph_hidden, decoder_hidden, decoder_input), -1)
        pgen = torch.sigmoid(self.linear_all(all_hidden))   # (bs,)
        num_class = output_projections.size(1)
        output_projections_probs = pgen * output_projections_probs
        # if random.random() < 0.01:
        #     print("pgen: ", pgen)
        for b in range(num_class):
            if b in self.vocab_to_idx.keys():
                output_projections_probs[:, b] += (1 - pgen.squeeze(1)) * probs[:, self.vocab_to_idx[b]].squeeze(1)
        return output_projections_probs, state

    @staticmethod
    def _get_loss(logits: torch.LongTensor, targets: torch.LongTensor, target_mask: torch.LongTensor) -> torch.Tensor:

        relevant_targets = targets[:, 1:].contiguous()
        relevant_mask = target_mask[:, 1:].contiguous()  # bs * decoding_step

        return my_sequence_cross_entropy_with_logits(logits.contiguous(), relevant_targets, relevant_mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        all_metrics.update(self.kd_metric.get_metric(reset=reset))
        all_metrics.update({"BLEU_avg": self.bleu_aver.get_metric(reset=reset)})
        all_metrics.update({"BLEU1": self.bleu1.get_metric(reset=reset)})
        # all_metrics.update({"dink1": self.dink1.get_metric(reset=reset)})
        # all_metrics.update({"dink2": self.dink2.get_metric(reset=reset)})
        return all_metrics

