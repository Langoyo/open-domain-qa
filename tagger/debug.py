# %%
"""
ADJ: adjective
ADP: adposition
ADV: adverb
AUX: auxiliary
CCONJ: coordinating conjunction
DET: determiner
INTJ: interjection
NOUN: noun
NUM: numeral
PART: particle
PRON: pronoun
PROPN: proper noun
PUNCT: punctuation
SCONJ: subordinating conjunction
SYM: symbol
VERB: verb
X: other
"""

# %%
import nltk
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
sentences = nltk.corpus.treebank.tagged_sents(tagset='universal')
print("Number of Tagged Sentences ",len(sentences))
print(sentences[0])
 
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# %%


word_to_idx = {'UNK':0}
tag_to_idx = {}
char_to_idx = {'UNK':0}


def prepare_words(seq,to_ix):
    idxs = []
    for w in seq:
        if w not in to_ix.keys():
            idxs.append(to_ix['UNK'])
        else:
            idxs.append(to_ix[w])
    return torch.tensor(idxs, dtype=torch.long,device=device)

        

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long,device=device)
def prepare_chars(sentence, to_ix):
    indexes = []
    for word in sentence:
        idxs = []
        for char in word:
            if char not in to_ix.keys():
                idxs.append(to_ix['UNK'])
            else:
                idxs.append(to_ix[char])
        indexes.append(idxs)
    max_length = max(len(row) for row in indexes)
    padded =   [row + [0] * (max_length - len(row))  for row in indexes]
    return torch.tensor(padded, dtype=torch.long,device=device)


for sentence in sentences:
    for word, tag in sentence:
        if word not in word_to_idx.keys():
            word_to_idx[word] = len(word_to_idx)
        if tag not in tag_to_idx.keys():
            tag_to_idx[tag] = len(tag_to_idx)
        for char in word:
            if char not in char_to_idx.keys():
                char_to_idx[char] = len(char_to_idx)

print(len(word_to_idx),len(tag_to_idx),len(char_to_idx))

# %%
print(tag_to_idx)
idx_to_tag = {}
confusion_matrix_labels = []
for key, index in tag_to_idx.items():
    idx_to_tag[index] = key
    confusion_matrix_labels.append(key)
print(idx_to_tag)


# %%
class Features():
    def fill(self, word_to_idx, tag_to_idx, char_to_idx):
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        self.char_to_idx = char_to_idx
        
        self.idx_to_tag = {}
        for key,index in self.tag_to_idx.items():
            self.idx_to_tag[index] = key
        
    def save(self):
        """save class as self.name.txt"""
        file = open('features.txt','wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self):
        """try load self.name.txt"""
        file = open('features.txt','rb')
        dataPickle = file.read()
        file.close()

features = Features()
features.fill(word_to_idx, tag_to_idx, char_to_idx)
features.save()

print(features)


# %%
data = []
for sentence in sentences:
    words = []
    tags = []
    for word, tag in sentence:
        words.append(word) 
        tags.append(tag)
    data.append((words,tags))

train_len = math.floor(len(data)*0.8)
val_len = math.floor(len(data)*0.1)
test_len = math.floor(len(data)*0.1)

train = data[:train_len]
validation = data[train_len:train_len+val_len]
test = data[train_len+val_len:]
print(len(train),len(validation),len(test))

# %%
# class LSTMTagger(nn.Module):

#     def __init__(self, embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim, vocab_size, char_vocab_size, tagset_size):
#         super(LSTMTagger, self).__init__()
#         self.hidden_dim = hidden_dim

#         self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

#         # The LSTM takes word embeddings as inputs, and outputs hidden states
#         # with dimensionality hidden_dim.
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

#         self.lstm_output_dropout = torch.nn.Dropout(1 - KEEP_DROPOUT)

#         # The linear layer that maps from hidden state space to tag space
#         self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        

#     def forward(self, sentence):
#         embeds = self.word_embeddings(sentence)
#         lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
#         lstm_out_dropped = self.lstm_output_dropout(lstm_out)
#         tag_space = self.hidden2tag(lstm_out_dropped.view(len(sentence), -1))
#         tag_scores = F.log_softmax(tag_space, dim=1)
#         return tag_scores


# %%
class char_LSTM(nn.Module):
    '''El Chapo'''
    def __init__(self, char_embedding_dim, char_hidden_dim, charset_size):
        super(char_LSTM, self).__init__()
        
        self.char_hidden_dim = char_hidden_dim
        self.char_embedding = nn.Embedding(charset_size, char_embedding_dim)
        self.lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)
        self.char_hidden = self.init_hidden()
        
    def init_hidden(self):
        ''' Intialize the hidden state'''
        return (torch.rand(1,1,self.char_hidden_dim),
               torch.rand(1,1,self.char_hidden_dim))
    
    def forward(self,single_word):
        ''' Return the final hidden state a.k.a char embedding(This encodes dense character features )'''
        char_embeds = self.char_embedding(single_word)
        self.char_hidden = self.init_hidden()
        _, self.char_hidden = self.lstm(char_embeds.view(len(single_word),1,-1))#,self.char_hidden)
        return self.char_hidden[0]

class LSTMTagger(nn.Module):
    '''GodFather'''
    def __init__(self, embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim, vocab_size, char_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.char_LSTM_embedding = char_LSTM(char_embedding_dim, char_hidden_dim, char_size)
        # note : LSTM input size is embedding_dim+char_hidden_dim to play nicely with concatenation
        self.lstm = nn.LSTM(embedding_dim+char_hidden_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        ''' Intialize the hidden state'''
        return (torch.rand(1,1,self.hidden_dim),
               torch.randn(1,1,self.hidden_dim))
    
    def concat_embeddings(self,some_word_embedding_tensor, some_character_embedding_tensor):
        ''' Concatenate the word embedding and character embedding into a single tensor. Do this for all words'''
        combo = []
        for w,c in zip(some_word_embedding_tensor,some_character_embedding_tensor):
            combo.append(torch.cat((w,c)))
        return torch.stack(combo)
    
    def forward(self, sentence, sentence_chars):
        word_embeds = self.word_embeddings(sentence)
        char_embeds = []
        for single_word_char in sentence_chars:
            # iterate through each word and append the character embedding to char_embeds
            char_embeds.append(torch.squeeze(self.char_LSTM_embedding(single_word_char)))
        # Concatenate the word embedding with the char embedding( i.e the hidden state from the char_LSTM for each word)
        word_char_embeds = self.concat_embeddings(word_embeds, char_embeds)
        lstm_out, self.hidden = self.lstm(word_char_embeds.view(len(sentence), 1, -1), (self.hidden[0].detach(),self.hidden[1].detach()))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence),-1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

# %%
WORD_EMBEDDING_DIM = 128
CHAR_EMBEDDING_DIM = 128
WORD_HIDDEN_DIM = 64
CHAR_HIDDEN_DIM = 64
KEEP_DROPOUT = 0.8


# %%
model = LSTMTagger(WORD_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, WORD_HIDDEN_DIM, CHAR_HIDDEN_DIM,
                   len(word_to_idx), len(char_to_idx), len(tag_to_idx))

model.to(device)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

import tqdm
def trainer(model, loss_function, optimizer, train):
    total_loss = 0
    predictions = []
    trues = []

    for sentence, tags in tqdm.tqdm(train):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        optimizer.zero_grad(set_to_none=True)


        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_words(sentence, word_to_idx)
        char_sequence = prepare_chars(sentence, char_to_idx)
        
        targets = prepare_sequence(tags, tag_to_idx)
        sentence_in.to(device)
        char_sequence.to(device)
        targets.to(device)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in, char_sequence)
        # print(tag_scores.size())
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        # tag_scores = torch.squeeze(tag_scores,0)
        # print(tag_scores.size())

        loss = loss_function(tag_scores, targets)
        total_loss += loss.item()

        _, indices = torch.max(tag_scores, 1)

        predictions = predictions + indices.tolist()
        trues = trues + targets.tolist()

        loss.backward()
        optimizer.step()
    return total_loss, predictions, trues
    

def tester(model, loss_function, test):
    total_loss = 0
    predictions = []
    trues = []

    for sentence, tags in tqdm.tqdm(test):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_words(sentence, word_to_idx)
        char_sequence = prepare_chars(sentence, char_to_idx)
        targets = prepare_sequence(tags, tag_to_idx)
        sentence_in.to(device)
        char_sequence.to(device)
        targets.to(device)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in,char_sequence)

        # tag_scores = torch.squeeze(tag_scores,0)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        
        loss = loss_function(tag_scores, targets)
        total_loss += loss.item()

        
        

        # Caluclating the accuracy of tags
        _, indices = torch.max(tag_scores, 1)

        predictions = predictions + indices.tolist()
        trues = trues + targets.tolist()
    
    return total_loss, predictions, trues
    
    


# %%
from sklearn.metrics import f1_score, accuracy_score
import tqdm
TOTAL_EPOCHS = 20
train_losses = []
validation_losses = []
accuracies_t = []
accuracies_v = []
f1s_t = []
f1s_v = []
for epoch in range(TOTAL_EPOCHS):
    train_loss, pred_t, true_t = trainer(model, loss_function, optimizer,train)
    val_loss, pred_v, true_v = tester(model, loss_function, validation)
    print(f"EPOCH: {epoch}")
    print(f"TRAIN LOSS: {train_loss}")
    print(f"VALIDATION LOSS: {val_loss}")
    accuracy_v = accuracy_score(true_v, pred_v)
    accuracy_t = accuracy_score(true_t, pred_t)
    f1_v = f1_score(true_v, pred_v, average='weighted')
    f1_t = f1_score(true_t, pred_t, average='weighted')
    # print(f"VAL F-1: {f1_score(true_v, pred_v, average='weighted')}")
    # print(f"VAL ACC: {accuracy}")

    f1s_t.append(f1_t)
    f1s_v.append(f1_v)
    accuracies_t.append(accuracy_t)
    accuracies_v.append(accuracy_v)
    train_losses.append(train_loss)
    validation_losses.append(val_loss)

# %%
test_loss, pred, true = tester(model, loss_function, test)
print(f"VALIDATION LOSS: {test_loss}")
print(f"VAL F-1: {f1_score(true, pred, average='weighted')}")
print(f"VAL ACC: {accuracy_score(true, pred)}")

# %%
import matplotlib.pyplot as plt
plt.title('Loss')
t, = plt.plot(train_losses, c="red", label ="train")
v, = plt.plot(validation_losses, c="blue", label ="validation")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend(handles=[t,v], title="title",loc=4, fontsize='small', fancybox=True)
plt.show()


plt.title('Accuracy')
t, = plt.plot(accuracies_t, c="red", label ="train")
v, = plt.plot(accuracies_v, c="blue", label ="validation")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend(handles=[t,v], title="title",loc=4, fontsize='small', fancybox=True)

plt.show()

plt.title('F1-score')
t, = plt.plot(f1s_t, c="red", label ="train")
v, = plt.plot(f1s_v, c="blue", label ="validation")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend(handles=[t,v], title="title",loc=4, fontsize='small', fancybox=True)


# %%
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


df_cm = pd.DataFrame(confusion_matrix(true, pred), index=confusion_matrix_labels, columns= confusion_matrix_labels)
plt.figure(figsize=(10,7))
# sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True,fmt='g', annot_kws={"size": 16}) # font size

plt.show()

# %%
torch.save(model.state_dict(), './model_file')

# %%
import pickle
def prepare_words(seq,to_ix):
    idxs = []
    for w in seq:
        if w not in to_ix.keys():
            idxs.append(to_ix['UNK'])
        else:
            idxs.append(to_ix[w])
    return torch.tensor(idxs, dtype=torch.long,device=device)


class Features():
    def fill(self, word_to_idx, tag_to_idx, char_to_idx):
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        self.char_to_idx = char_to_idx
        
        self.idx_to_tag = {}
        for key,index in self.tag_to_idx.items():
            self.idx_to_tag[index] = key
        
    def save(self):
        """save class as self.name.txt"""
        file = open('features.txt','wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self):
        """try load self.name.txt"""
        file = open('features.txt','rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)
features = Features()
features.load()


# %%
import nltk
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = 'cpu'

WORD_EMBEDDING_DIM = 128
CHAR_EMBEDDING_DIM = 128
WORD_HIDDEN_DIM = 64
CHAR_HIDDEN_DIM = 64
KEEP_DROPOUT = 0.8


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim, vocab_size, char_vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        self.lstm_output_dropout = torch.nn.Dropout(1 - KEEP_DROPOUT)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        lstm_out_dropped = self.lstm_output_dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out_dropped.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# %%



model = LSTMTagger(WORD_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, WORD_HIDDEN_DIM, CHAR_HIDDEN_DIM,
                   len(features.word_to_idx), len(features.char_to_idx), len(features.tag_to_idx))
model.load_state_dict(torch.load('./model_file'))
model.to(device)


# %%
def predict(model, query):
    query = nltk.word_tokenize(query)
    model.zero_grad()

    sentence_in = prepare_words(query, features.word_to_idx)
    
    tag_scores = model(sentence_in)

    _, indices = torch.max(tag_scores, 1)

    indices = indices.tolist()

    preds = [features.idx_to_tag[x] for x in indices]
    print(preds)
    print(query)
    
predict(model,'When did Messi win the champions league?')


