from torch import nn
from transformers import (
    BertPreTrainedModel,
    BertModel,
    RobertaPreTrainedModel,
    RobertaModel,
    ElectraPreTrainedModel,
    ElectraModel,
)


class RetrievalBERTEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(RetrievalBERTEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs[1]

        return pooled_output


class RetrievalRoBERTaEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RetrievalRoBERTaEncoder, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs[1]

        return pooled_output


class RetrievalELECTRAEncoder(ElectraPreTrainedModel):
    def __init__(self, config):
        super(RetrievalELECTRAEncoder, self).__init__(config)

        self.electra = ElectraModel(config)
        self.init_weights()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.electra(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        first_token_tensor = outputs[0][:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output
