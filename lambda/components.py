import torch
# from transformers import BertForQuestionAnswering, AutoTokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer
from transformers import pipeline

# from gensim.summarization.bm25 import BM25
import wikipedia
from wikipedia import DisambiguationError
from utils import *
import torch.nn as nn
import nltk
import torch.nn.functional as F


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
        return (torch.rand(1,1,self.char_hidden_dim, device='cpu'),
               torch.rand(1,1,self.char_hidden_dim, device='cpu'))
    
    def forward(self,single_word):
        ''' Return the final hidden state a.k.a char embedding(This encodes dense character features )'''
        char_embeds = self.char_embedding(single_word)
        self.char_hidden = self.init_hidden()
        _, self.char_hidden = self.lstm(char_embeds.view(len(single_word),1,-1),self.char_hidden)
        return self.char_hidden[0]

class LSTMTagger(nn.Module):
    '''GodFather'''
    def __init__(self, embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim, vocab_size, char_size, tagset_size,word_num_layers):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.char_LSTM_embedding = char_LSTM(char_embedding_dim, char_hidden_dim, char_size)
        # note : LSTM input size is embedding_dim+char_hidden_dim to play nicely with concatenation
        self.lstm = nn.LSTM(embedding_dim+char_hidden_dim, hidden_dim,num_layers = word_num_layers)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.word_num_layers = word_num_layers
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        ''' Intialize the hidden state'''
        return (torch.rand(self.word_num_layers,1,self.hidden_dim, device='cpu'),
               torch.randn(self.word_num_layers,1,self.hidden_dim, device='cpu'))
    
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
        lstm_out, self.hidden = self.lstm(word_char_embeds.view(len(sentence), 1, -1), (self.hidden[0].detach(), self.hidden[1].detach()))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence),-1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class LSTMFeatures():
    def __init__(self, features_file):
        self.features_file = features_file
    def fill(self, word_to_idx, tag_to_idx, char_to_idx):
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        self.char_to_idx = char_to_idx
        
        self.idx_to_tag = {}
        for key,index in self.tag_to_idx.items():
            self.idx_to_tag[index] = key
        
    def save(self):
        """save class as self.name.txt"""
        file = open(self.features_file,'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self):
        """try load self.name.txt"""
        file = open(self.features_file,'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)


def prepare_words(seq,to_ix):
    idxs = []
    for w in seq:
        if w not in to_ix.keys():
            idxs.append(to_ix['UNK'])
        else:
            idxs.append(to_ix[w])
    return torch.tensor(idxs, dtype=torch.long,device='cpu')




class Tagger():
    def __init__(self, language):
        self.features_file = 'tagger/features.txt'
        self.model_file = 'tagger/model_file'

        if language is 'es':
            self.features_file = 'tagger/features_es.txt'
            self.model_file = 'tagger/model_file_es'
            
        self.features = LSTMFeatures(self.features_file)
        self.features.load()

        self.model = LSTMTagger(300, 100, 512, 128,
                   len(self.features.word_to_idx), len(self.features.char_to_idx), len(self.features.tag_to_idx))
        self.model.load_state_dict(torch.load(self.model_file))
        self.model.to('cpu')

    def prepare_words(self,seq,to_ix):
        idxs = []
        for w in seq:
            if w not in to_ix.keys():
                idxs.append(to_ix['UNK'])
            else:
                idxs.append(to_ix[w])
        return torch.tensor(idxs, dtype=torch.long,device='cpu')

    def prepare_sequence(self,seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long,device='cpu')

    def prepare_chars(self,sentence, to_ix):
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
        return torch.tensor(padded, dtype=torch.long,device='cpu')


    def predict(self, query):
        query = nltk.word_tokenize(query)
        self.model.zero_grad()

        sentence_in = self.prepare_words(query, self.features.word_to_idx)
        char_sequence = self.prepare_chars(query, self.features.char_to_idx)
        tag_scores = self.model(sentence_in,char_sequence)
        
        _, indices = torch.max(tag_scores, 1)

        indices = indices.tolist()

        preds = [self.features.idx_to_tag[x] for x in indices]
        return preds

class ClassificationModel(nn.Module):
    
    def __init__(self, vocab_size,embedding_dim, hidden_dim):
        super().__init__()
        print(vocab_size)
        # self.embedded_layer = nn.Embedding(vocab_size,embedding_dim=embedding_dim)
        self.e = torch.nn.Linear(vocab_size, 16)
        self.h1 = torch.nn.Linear(64,64)
        self.h2 = torch.nn.Linear(64,64)
        self.s = torch.nn.Linear(16,1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        

    def forward(self, x):
        x = self.e(x)        
        x = self.relu(x)
        # x = self.h1(x)
        # x = self.relu(x)
        # x = self.h2(x)
        # x = self.relu(x)
        x = self.s(x)
        output = self.sigmoid(x)
        
        return output

from nltk.tokenize import regexp_tokenize
import numpy as np
import pickle
import torch

# Here is a default pattern for tokenization, you can substitue it with yours
default_pattern =  r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """

def tokenize(text, pattern = default_pattern):
    """Tokenize senten with specific pattern
    
    Arguments:
        text {str} -- sentence to be tokenized, such as "I love NLP"
    
    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
    
    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    text = str(text)
    text = text.lower()
    return str(regexp_tokenize(text, pattern))


class FeatureExtractor(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass




class UnigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        # dictionary that maps unigrams into their indexes by order of appearance
        self.unigram = {}
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue
                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        # Feature has the size of the number of distinct words received
        feature = np.zeros(len(self.unigram))
        # We go through every word in the text
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                # If it contained in the unigram,
                # It is a feature of the new text.
                # We search for the index of the word in the text and use that index in the feature as well
                # Then, we increase the number of occurrences of that word.
                # Seems like the bag of words seen in class.
                feature[self.unigram[text[i].lower()]] += 1
        
        return feature.tolist()
    
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        
        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        # Same as previous method but at sentences level
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        
        
        return np.array(features)

    def indexesToWords(self, words):
        for word, index in self.unigram.items():
            if index in words:
                print(word)
    
    def save(self):
        """save class as self.name.txt"""
        file = open('./drive/MyDrive/TFG/language/feature_extractor.txt','wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self,path):
        """try load self.name.txt"""
        file = open(path,'rb')
        dataPickle = file.read()
        file.close()

        self.__dict__ = pickle.loads(dataPickle)

class LanguageID():
    def __init__(self):
        self.feature_extractor = UnigramFeature()
        self.feature_extractor.load('./language_detector/feature_extractor.txt')
        self.model = ClassificationModel(241,0,0)
        self.model.load_state_dict(torch.load('language_detector/model_file'))

    def predict(self, query):
        query = tokenize(query)
        query = self.feature_extractor.transform(query)
        query = torch.tensor(query)

        self.model.zero_grad()    
        result = self.model(query)
        result = round(result.item())
        if result == 0:
            return'en'
        else:
            return'es'


class QueryGenerator():
    def __init__(self,language = 'en'):
        # Model Used
        self.model = Tagger(language)
        # Relevant POS tags that could be useful for a query
        self.relevant = ['PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ']

    def generate_query(self,query):
        """Generates query words basded on an input query"""
        # We obtain tokens and their tags form the query
        analysis = self.model.predict(query)
        # We keep the tokens that could eb relevant and also use the previous token
        query = [token.text for token in analysis if token.pos_ in self.relevant] + [query]
        return [query]

class DocumentExtractor():
    def __init__(self, language = "en"):
        wikipedia.set_lang(language)


    def get_documents(self, queries):
        """Searches the documents from having introduced a list of query words/sentences"""
        for query in queries:
            suggestions = wikipedia.search(query)
        library = {
            'titles':[],
            'contents':[],
            'urls':[]
            }
        for suggestion in suggestions:
            try:
                text = wikipedia.page(suggestion,auto_suggest=False)
                library["titles"].append(text.title) 
                library["contents"].append(text.content)
                library["urls"].append(text.url)
                
            except DisambiguationError as e:
                # text = wikipedia.page(e.options[0],auto_suggest=False)
                # library["titles"].append(text.title) 
                # library["contents"].append(text.content)
                # library["urls"].append(text.url)
                pass
                
        return library


class PassageRanker():
    """From a set of documents, returns the set of ranked passages"""    
            
    def text_splitter(self,text,n=175):
        """Splits a document into passages of n words (to avoid Reader errors wiht # of tokens)"""
        pieces = text.split()
        return list((" ".join(pieces[i:i+n]) for i in range(0, len(pieces),n)))
    
    def results_creator(self,library):
        """
        Creates the final results data structure which contains score, title, content of the passage and url
        It also returns the list with all passages created
        """
        results = []
        passages = []
        for i in range(len(library['contents'])):
            new_passages = self.text_splitter(library['contents'][i])
            passages += new_passages
            for text in new_passages:
                results.append([0, library['titles'][i], text, library['urls'][i]])
        return results,passages

    def rank(self, library, original_query, method = 'BM25'):
        """
        The receives a set of texts, the input query and the method to rank them
        For the moment methods are: 'BM25' and 'TF-IDF'

        It returns a list of passages with the following structure
            {'score':, 'title':, 'content':, 'url':}
        And ordered by score(likelihood of containing the answer)

        """
        results,passages = self.results_creator(library)
        
        # if method == 'BM25':
        #     corpus = [tokenize(p) for p in passages]
        #     bm25 = gensim.summarization.bm25.BM25(corpus)
        #     question = tokenize(original_query)
        #     scores = bm25.get_scores(question)
        #     pairs = [(score, i) for i, score in enumerate(scores)]
        #     pairs.sort(reverse=True)
        
        # if method == 'BERT':
        #     model = SentenceTransformer('all-MiniLM-L6-v2',device='cpu')
        #     embeddings1 = model.encode(passages, convert_to_tensor=True)
        #     embeddings2 = model.encode(original_query, convert_to_tensor=True)
        #     score = util.cos_sim(embeddings1, embeddings2)
        #     score = score.tolist()
        
        # else:

        feature = SPARSE(bigram=False)
        feature.fit(passages)
        # question = tokenize(original_query)
        # feature.fit(corpus)#+[query])
        # tf_idfs = feature.transform_list(corpus)
        # quer = feature.transform_list([original_query])
        score = feature.get_scores(original_query, passages, method=method)


        

        for i in range(len(results)):
            results[i][0] = score[i]#cosine_similarity(quer[0],tf_idfs[i])
        results.sort(key = lambda x:x[0], reverse=True)

        
        return results
        

class DocumentReader():
    def __init__(self, lang):
        if (lang=='es'):
            self.qa_pipeline = pipeline(
                "question-answering",
                model="./bert-qa-es",
                tokenizer="./bert-qa-es")


        else:


            self.qa_pipeline = pipeline(
                "question-answering",
                model="./bert-qa-en",
                tokenizer="./bert-qa-en"
)

            

        
        
        

    def answer_question(self,question, answer_text):
        '''
        Takes a `question` string and an `answer_text` string (which contains the
        answer), and identifies the words within the `answer_text` that are the
        answer. Prints them out.
        '''
        # # ======== Tokenize ========
        # # Apply the tokenizer to the input text, treating them as a text-pair.
        # input_ids = self.tokenizer.encode(question, answer_text)

        # # Report how long the input sequence is.

        # # ======== Set Segment IDs ========
        # # Search the input_ids for the first instance of the `[SEP]` token.
        # sep_index = input_ids.index(self.tokenizer.sep_token_id)

        # # The number of segment A tokens includes the [SEP] token istelf.
        # num_seg_a = sep_index + 1

        # # The remainder are segment B.
        # num_seg_b = len(input_ids) - num_seg_a

        # # Construct the list of 0s and 1s.
        # segment_ids = [0]*num_seg_a + [1]*num_seg_b

        # # There should be a segment_id for every input token.
        # assert len(segment_ids) == len(input_ids)

        # # ======== Evaluate ========
        # # Run our example through the model.
        # outputs = self.model(torch.tensor([input_ids]), # The tokens representing our input text.
        #                 token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
        #                 return_dict=True) 

        # start_scores = outputs.start_logits
        # end_scores = outputs.end_logits

        # # ======== Reconstruct Answer ========
        # # Find the tokens with the highest `start` and `end` scores.
        # answer_start = torch.argmax(start_scores)
        # answer_end = torch.argmax(end_scores)

        # # Get the string versions of the input tokens.
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # # Start with the first token.
        # answer = tokens[answer_start]

        # # Select the remaining answer tokens and join them with whitespace.
        # for i in range(answer_start + 1, answer_end + 1):
            
        #     # If it's a subword token, then recombine it with the previous token.
        #     if tokens[i][0:2] == '##':
        #         answer += tokens[i][2:]
            
        #     # Otherwise, add a space then the token.
        #     else:
        #         answer += ' ' + tokens[i]
        answer = self.qa_pipeline({
                'context': answer_text,
                'question': question

            })

        return answer['answer']

    def answer(self,question,results,n_results=10):
        answers = []
        for i in range(min(len(results),n_results)):
            try:
                answer = self.answer_question(question,results[i][2])
                answers.append({'score':results[i][0],
                            'title':results[i][1],
                            'content':results[i][2],
                            'url':results[i][3],
                            'answer':answer})
            except RuntimeError:
                answers.append({'score':results[i][0],
                            'title':results[i][1],
                            'content':results[i][2],
                            'url':results[i][3],
                            'answer':'An error occurred'})

        return answers

        



        