import re
import pickle
import torch
import torch.nn as nn

MAX_LEN = 512

LABELS = ["A","B","C","D","E"]


class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=200,
        hidden_dim=256,
        num_layers=1,
        num_classes=5,
        dropout=0.4,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):

        x = self.embedding(x)

        outputs,(hidden,cell)=self.lstm(x)

        forward_hidden=hidden[-2]
        backward_hidden=hidden[-1]

        hidden=torch.cat((forward_hidden,backward_hidden),dim=1)

        hidden=self.dropout(hidden)

        return self.fc(hidden)


with open("word2idx.pkl","rb") as f:
    word2idx=pickle.load(f)


def tokenize(text):

    text=text.lower()

    text=re.sub(r"[^a-z0-9\s]"," ",text)

    return text.split()


def encode(text):

    ids=[]

    for token in tokenize(text):

        ids.append(word2idx.get(token,1))

    ids=ids[:MAX_LEN]

    if len(ids)<MAX_LEN:

        ids += [0]*(MAX_LEN-len(ids))

    return ids


device=torch.device("cpu")

model=BiLSTMClassifier(len(word2idx))

model.load_state_dict(
    torch.load(
        "best_bilstm_model.pth",
        map_location=device,
    )
)

model.eval()


def predict(question,A,B,C,D,E):

    text=f"""
Question: {question}
A: {A}
B: {B}
C: {C}
D: {D}
E: {E}
"""

    x=torch.tensor(
        [encode(text)],
        dtype=torch.long,
    )

    with torch.no_grad():

        logits=model(x)

        probs=torch.softmax(logits,dim=1)[0]

    values,indices=torch.topk(probs,3)

    result=[]

    for p,i in zip(values,indices):

        result.append((LABELS[i],float(p)))

    return result
