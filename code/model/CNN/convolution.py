import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class RobertaConv(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaConv, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.init_weights()

        self.conv1d_kernel1 = nn.Conv1d(config.hidden_size, config.hidden_size // 3, kernel_size=1)
        self.conv1d_kernel3 = nn.Conv1d(config.hidden_size, config.hidden_size // 3, kernel_size=3, padding=1)
        self.conv1d_kernel5 = nn.Conv1d(config.hidden_size, config.hidden_size // 3, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(config.hidden_size // 3 * 3, 2, bias=True)

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
                return_dict=None
                ):

        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict,
                               )

        sequence_output = outputs[0]  # (batch_size, max_seq_length, hidden_size)
        conv_input = sequence_output.transpose(1, 2)  # Conv 연산을 위한 Transpose (batch_size, hidden_size, max_seq_length)
        conv_output1 = F.relu(self.conv1d_kernel1(conv_input))  # Conv 연산의 결과 (batch_size, conv_out_cnannel, max_seq_length)
        conv_output2 = F.relu(self.conv1d_kernel3(conv_input))  # Conv 연산의 결과 (batch_size, conv_out_cnannel, max_seq_length)
        conv_output3 = F.relu(self.conv1d_kernel5(conv_input))  # Conv 연산의 결과 (batch_size, conv_out_cnannel, max_seq_length)
        concat_output = torch.cat((conv_output1, conv_output2, conv_output3), dim=1)  # Concatenation (batch_size, conv_out_cnannel * 3, max_seq_length)
        concat_output = concat_output.transpose(1, 2)  # Linear 연산을 위한 Transpose (batch_size, max_seq_length, conv_out_channel * 3)
        concat_output = self.dropout(concat_output)
        logits = self.linear(concat_output)  # (batch_size, max_seq_length, 2)

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

        # if not return_dict:
        #     output = (start_logits, end_logits) + outputs[2:]
        #     return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

