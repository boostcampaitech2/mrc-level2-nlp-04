from transformers import AutoModel, PreTrainedModel


class RetrievalEncoder(PreTrainedModel):
    def __init__(self, model_name, model_config):
        super(RetrievalEncoder, self).__init__(model_config)

        self.model_name = model_name
        self.model_config = model_config

        self.encoder = AutoModel.from_pretrained(model_name, config=model_config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)

        if 'electra' in self.model_name:
            pooled_output = outputs[0][:, 0, :]
        else:
            pooled_output = outputs[1]
        return pooled_output
