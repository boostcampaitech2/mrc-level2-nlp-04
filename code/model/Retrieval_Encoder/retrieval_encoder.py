from transformers import AutoModel, PreTrainedModel
import torch.nn as nn
import torch
from torch.cuda.amp import autocast

class RetrievalEncoder(PreTrainedModel):
    def __init__(self, model_name, model_config):
        super(RetrievalEncoder, self).__init__(model_config)

        self.roberta = AutoModel.from_pretrained(model_name, config=model_config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)

        pooled_output = outputs[1]
        return pooled_output

class BiLSTM_RetrievalEncoder(PreTrainedModel):
    def __init__(self, model_name, model_config):
        super(BiLSTM_RetrievalEncoder, self).__init__(model_config)

        self.model = AutoModel.from_pretrained(model_name, config=model_config)
        self.hidden_dim = model_config.hidden_size  # roberta hidden dim = 1024

        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=2, dropout=0.2,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Conv1d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1)

    @autocast()
    def forward(self, input_ids, attention_mask):
        # BERT output= (16, 244, 1024) (batch, seq_len, hidden_dim)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

        # LSTM last hidden, cell state shape : (2, 244, 1024) (num_layer, seq_len, hidden_size)
        hidden, (last_hidden, last_cell) = self.lstm(output)

        # (16, 1024) (batch, hidden_dim)
        cat_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        logits = self.fc(cat_hidden.view(-1,self.hidden_dim*2,1))
        return logits.squeeze()