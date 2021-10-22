from transformers import AutoModel, PreTrainedModel


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
