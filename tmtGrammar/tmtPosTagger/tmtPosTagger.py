
import sys
sys.path.append("../")

from artifact import *

from transformers import (
            AutoModel, 
            BertForTokenClassification,
            BertConfig, 
            )


class TMTPosTaggingModel(BertForTokenClassification):
    def __init__(self, pretrained_bert_ckp, num_labels):
        
        pretrained_bert = AutoModel.from_pretrained(pretrained_bert_ckp, add_pooling_layer=False)
        config = BertConfig()
        config = pretrained_bert.config
        config.num_labels = num_labels
        config.id2label = ID2LABEL
        config.label2id = LABEL2ID
        
        super().__init__(config)
        self.bert = pretrained_bert
