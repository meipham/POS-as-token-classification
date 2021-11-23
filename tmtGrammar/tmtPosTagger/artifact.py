
PRETRAINED_BERT_DIR = "D:/tmt/tmtGrammar/pho/phobert-base" 
PRETRAINED_BERT_URL = "vinai/phobert-base"

TAGSET = {  'V': 0,
            'T': 1,
            'Ch': 2,
            'P': 3,
            'Np': 4,
            'Cc': 5,
            'Nb': 6,
            'Y': 7,
            'Vb': 8,
            'N': 9,
            'E': 10,
            'Ny': 11,
            'M': 12,
            'R': 13,
            'A': 14,
            'I': 15,
            'C': 16,
            'L': 17,
            'Nu': 18,
            'Nc': 19}

IDS = list(TAGSET.keys())
LABELS = list(TAGSET.values())

ID2LABEL = dict(zip(IDS, LABELS))
LABEL2ID = dict(zip(LABELS, IDS))