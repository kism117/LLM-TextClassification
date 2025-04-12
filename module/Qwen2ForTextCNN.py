
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

import inspect

from transformers.cache_utils import (
    Cache
)

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

from transformers.modeling_outputs import (
    SequenceClassifierOutputWithPast,
)

from transformers.models.qwen2.modeling_qwen2 import (
    QWEN2_START_DOCSTRING,
    QWEN2_INPUTS_DOCSTRING,
    Qwen2PreTrainedModel,
    Qwen2Model,
)



logger = logging.get_logger(__name__)

@add_start_docstrings(
    """
    The Qwen2 Model transformer with a sequence classification head on top (linear layer).

    [`Qwen2ForTextCNN`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    QWEN2_START_DOCSTRING,
)
class Qwen2ForTextCNN(Qwen2PreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        self.loss_CE =  torch.nn.CrossEntropyLoss()
        

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        

        #print("#output_hidden_states ",output_hidden_states)
        #print("#transformer_outputs ", type(transformer_outputs), " ", len(transformer_outputs))

        #all_hidden_states = transformer_outputs[1]
        #print("#all_hidden_states ",type(all_hidden_states)," ",len(all_hidden_states))
        #for i in range(len(all_hidden_states)):
        #    print("#all_hidden_states_",i," ",type(all_hidden_states[i])," ",all_hidden_states[i].shape)
        # [2,512,1536]
        #print(type(transformer_outputs[1][-1]))
        #print(transformer_outputs[1][-1].shape)
        #hidden_states = transformer_outputs[0]
        #hidden_states = transformer_outputs[1][-1]+transformer_outputs[0]

        #print(len(transformer_outputs[1]))
        #print(xxx)
        
        hidden_states = torch.cat([transformer_outputs[0].unsqueeze(1),transformer_outputs[1][14].unsqueeze(1),transformer_outputs[1][1].unsqueeze(1)],dim=1)
        
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        #pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        pooled_logits = logits
        loss = None
        if labels is not None:
            loss = self.loss_CE(logits, labels)

            # 不使用 默认的 self.loss_function
            #loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)
            
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
