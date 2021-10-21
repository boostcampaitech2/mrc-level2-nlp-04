from transformers.models.roberta.modeling_roberta import RobertaForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers import AutoConfig, AutoModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

# 현재는 roberta 한정이므로 추후에 importlib를 사용해서 사전에 만들어 놓은 mapping으로 로딩해야 합니다.

class ModelAttachedLSTM(RobertaForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)

        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size,
                            num_layers=2,
                            batch_first=False,
                            bidirectional=True,)
        self.qa_outputs = nn.Linear(config.hidden_size*2,
                                    config.num_labels,)
        self.roberta = AutoModel.from_pretrained("klue/roberta-small")

    def forward(
            self,
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
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        sequence_output = outputs[0]
        # print("1 : ", sequence_output.size())

        hidden, (last_hidden, last_cell) = self.lstm(sequence_output)
        # hidden = self.conv(sequence_output)

        # print("4 : ", last_hidden.size())
        # print("5 : ", last_hidden[0].size())
        # print("6 : ", last_cell.size())
        # print("7 : ", last_cell[0].size())
        # print("8 : ", hidden.size())
        # print("9 : ", hidden[0].size())
        # sequence_output = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        # print("2 : ", sequence_output.size())
        logits = self.qa_outputs(hidden)

        # logits = self.qa_outputs(sequence_output)
        # print("3 : ", logits.size())
        # exit()
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