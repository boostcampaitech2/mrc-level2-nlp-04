import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaPreTrainedModel, RobertaModel


class RobertaConv(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaConv, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.init_weights()

        self.conv1d_kernel1 = nn.Conv1d(config.hidden_size, 1024, kernel_size=1)
        self.conv1d_kernel3 = nn.Conv1d(config.hidden_size, 1024, kernel_size=3, padding=1)
        self.conv1d_kernel5 = nn.Conv1d(config.hidden_size, 1024, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(1024 * 3, 2, bias=True)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                start_positions=None,
                end_postions=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds,
                               start_positions=start_positions,
                               end_postions=end_postions,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict,
                               )

        sequence_output = outputs[0]  # (batch_size, max_seq_length, hidden_size)
        conv_input = sequence_output.transpose(1, 2)  # Conv 연산을 위한 Transpose (batch_size, hidden_size, max_seq_length)
        conv_output1 = F.relu(self.conv1d_kernel1(conv_input))  # Conv 연산의 결과 (batch_size, num_conv_filter, max_seq_length)

