import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class RobertaQAConv(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaQAConv, self).__init__(config)

        self.sep_token_id = 2  # roberta model 의 tokenizer 는 sep_token_id 가 2 임
        self.roberta = RobertaModel(config)
        self.query_drop_out = nn.Dropout(0.1)
        self.query_layer = nn.Linear(config.hidden_size * 50, config.hidden_size, bias=True)
        self.query_classify_layer = nn.Linear(config.hidden_size, 6, bias=True)
        self.key_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.conv1d_kernel1 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1)
        self.conv1d_kernel3 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1)
        self.conv1d_kernel5 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=5, padding=2)
        self.drop_out = nn.Dropout(0.3)
        self.classify_layer = nn.Linear(config.hidden_size * 3, 2, bias=True)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                start_positions=None,
                end_positions=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # (batch_size, max_seq_length, hidden_size)

        if not token_type_ids:
            token_type_ids = self.make_token_type_ids(input_ids)

        query = sequence_output * (token_type_ids.unsqueeze(dim=-1) == 0)  # 전체 text 중 query 에 해당하는 Embeded Vector 만 남김
        query = self.query_drop_out(F.relu(query))  # Activation Function 및 Dropout Layer 통과
        query = query[:, :50, :]  # 질문에 해당하는 Embedding 만 남김 (batch_size, 50, hidden_size)
        query = query.reshape((sequence_output.shape[0], 1, -1))  # Token 의 Embedding 을 Hidden Dim 축으로 Concat 함 (batch_size, 1, 50 * hidden_size)
        query = self.query_layer(query)  # Dense Layer 를 통과 시킴. (batch_size, 1, hidden_size)
        query_logits = F.softmax(self.query_classify_layer(query.squeeze(1)), dim=-1)  # Query 의 종류를 예측하는 Branch (batch_size, 6)

        key = sequence_output * (token_type_ids.unsqueeze(dim=-1) == 1)  # 전체 text 중 context 에 해당하는 Embeded Vector 만 남김
        key = self.key_layer(key)  # (batch_size, max_seq_length, hidden_size)
        attention_rate = torch.matmul(key, torch.transpose(query, 1, 2))  # Context 의 Value Vector 와 Question 의 Query Vector 를 사용 (batch_size, max_seq_length, 1)
        attention_rate = attention_rate / key.shape[-1]**0.5  # hidden_size 의 표준편차로 나눠줌 (batch_size, max_seq_length, 1)
        attention_rate = attention_rate / 10  # Temperature 로 나눠줌. (batch_size, max_seq_length, 1)
        attention_rate = F.softmax(attention_rate, dim=1)  # softmax 를 통과시켜서 확률값으로 변경해, Question 과 Context 의 Attention Rate 를 구함 (batch_size, max_seq_length, 1)

        value = self.value_layer(sequence_output)  # (batch_size, max_seq_length, hidden_size)
        value = value * attention_rate  # Attention Rate 를 활용해서 Output 값을 변경함  (batch_size, max_seq_length, hidden_size)

        conv_input = value.transpose(1, 2)  # Convolution 연산을 위해 Transpose (batch_size, hidden_size, max_seq_length)
        conv_output1 = F.relu(self.conv1d_kernel1(conv_input))  # Conv 연산의 결과 (batch_size, conv_out_cnannel, max_seq_length)
        conv_output2 = F.relu(self.conv1d_kernel3(conv_input))  # Conv 연산의 결과 (batch_size, conv_out_cnannel, max_seq_length)
        conv_output3 = F.relu(self.conv1d_kernel5(conv_input))  # Conv 연산의 결과 (batch_size, conv_out_cnannel, max_seq_length)
        concat_output = torch.cat((conv_output1, conv_output2, conv_output3), dim=1)  # Concatenation (batch_size, conv_out_channel * 3, max_seq_length)
        concat_output = concat_output.transpose(1, 2)  # Dense Layer 에 입력을 위해 Transpose (batch_size, max_seq_length, conv_out_channel * 3)
        concat_output = self.drop_out(concat_output)  # dropout 통과
        logits = self.classify_layer(concat_output)  # Classifier Layer 를 통해 최종 Logit 을 얻음 (batch_size, max_seq_length, 2)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def make_token_type_ids(self, input_ids):
        token_type_ids = []
        for i, input_id in enumerate(input_ids):
            sep_idx = np.where(input_id.cpu().numpy() == self.sep_token_id)
            token_type_id = [0] * sep_idx[0][0] + [1] * (len(input_id) - sep_idx[0][0])
            token_type_ids.append(token_type_id)
        return torch.tensor(token_type_ids).cuda()
