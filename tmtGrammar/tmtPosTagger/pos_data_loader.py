import pandas as pd
from transformers import AutoTokenizer
import datasets 

from artifact import *


features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "tokens": datasets.Sequence(datasets.Value("string")),
        "pos_tags": datasets.Sequence(
            datasets.features.ClassLabel(
                names=list(TAGSET.keys()),
                id=list(TAGSET.values())
            )
        ),
    }
)

def create_json(intxt, outjson):
    
    df = pd.DataFrame(columns=["id", "tokens", "pos_tags"])

    with open(intxt, "r", encoding="utf-8") as f:
        id = 0
        tokens, pos_tags = [], []

        for line in f: 
            line = line.strip()
            if line == "": 
                if len(tokens) > 0:
                    assert(len(tokens)==len(pos_tags))
                    df = df.append({"id": id, "tokens": tokens, "pos_tags": pos_tags}, ignore_index=True)
                    id=id+1
                    tokens, pos_tags = [], []

            else:
                line = line.split("\t")
                if len(line) != 2:
                    continue
                else:
                    token, pos_tag = line[0], line[1]
                    tokens.append(token)
                    pos_tags.append(TAGSET[pos_tag.capitalize()])

    df.to_json(outjson, orient='records', force_ascii=False, lines=True)

tokenizer = AutoTokenizer.from_pretrained(
    PRETRAINED_BERT_DIR,
    use_fast=False
)


def load_pos_dataset(data_files, padding, max_length):
    dataset = datasets.load_dataset("json", data_files=data_files, features=features)

    def preprocess_function(examples):
        tokenize_ids = tokenizer(examples["tokens"], max_length=max_length, truncation=True, padding=padding, is_split_into_words=True)

        vocab_ids = tokenizer.convert_tokens_to_ids(examples["tokens"])
        unk_indices = [idx for idx, id in enumerate(vocab_ids) if id==tokenizer.unk_token_id]
        
        # Aligning label ids according to word position
        label_ids = [-100] + examples["pos_tags"]
        for idx in unk_indices:
            unk_word_tok_len = tokenizer(examples["tokens"][idx], return_token_type_ids=False, return_attention_mask=False, add_special_tokens=False, return_length=True)["length"]
            label_ids = label_ids[:idx+1] + [-100]*(unk_word_tok_len-1) + label_ids[idx+1:]
        
        # Padding label ids list
        if len(label_ids) < max_length:
            label_ids = label_ids + [-100]*(max_length-len(label_ids))
        else:
            label_ids = label_ids[:max_length]


        tokenize_ids["labels"] = label_ids
        return tokenize_ids


    dataset = dataset.map(preprocess_function)

    columns_to_return = ['input_ids', 'labels', 'attention_mask']
    dataset.set_format(columns=columns_to_return)
    return dataset