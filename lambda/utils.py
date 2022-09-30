from nltk.tokenize import regexp_tokenize
import numpy as np
import math

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
    text = text.lower()
    return regexp_tokenize(text, pattern)


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
        
        return feature
    
    
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
        
    


class BigramFeature(FeatureExtractor):
    """Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self):
        self.bigram = {}
    def fit(self, text_set):
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])-1):
                merged_words = (text_set[i][j]+text_set[i][j+1]).lower()
                if merged_words not in self.bigram:
                    self.bigram[merged_words] = index
                    index += 1
                else:
                    continue

    def transform(self, text):
        # Feature has the size of the number of distinct words received
        feature = np.zeros(len(self.bigram))
        # We go through every word in the text
        for i in range(0, len(text)-1):
            merged_words = (text[i].lower()+text[i+1].lower())
            if merged_words in self.bigram:
                # If it contained in the unigram,
                # It is a feature of the new text.
                # We search for the index of the word in the text and use that index in the feature as well
                # Then, we increase the number of occurrences of that word.
                # Seems like the bag of words seen in class.
                feature[self.bigram[merged_words]] += 1
        
        return feature
    def transform_list(self, text_set):
        # Add your code here!
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)

    def indexesToWords(self, words):
        for word, index in self.unigram.items():
            if index in words:
                print(word)

    


class TFIDF(FeatureExtractor):
    """customized feature extractor,nTF-IDF (term frequency / inverse document frequency)
    """
    def __init__(self,bigram=False):
        # Add your code here!
        self.unigram = {}
        self.bigram = {}
        self.mode = bigram


    def fit(self, text_set):
        # Creates unigrams dict
        # for i in range(len(text_set)):
        #     print(text_set[i])
        #     text_set[i] = tokenize(text_set[i])
        index = 0
        if not self.mode:
            for i in range(0, len(text_set)):
                for j in range(0, len(text_set[i])):
                    if text_set[i][j].lower() not in self.unigram:
                        self.unigram[text_set[i][j].lower()] = index
                        index += 1
                    else:
                        continue
        else:
            index = 0
            for i in range(0, len(text_set)):
                for j in range(0, len(text_set[i])-1):
                    merged_words = (text_set[i][j]+text_set[i][j+1]).lower()
                    if merged_words not in self.bigram:
                        self.bigram[merged_words] = index
                        index += 1
                    else:
                        continue      
    def transform(self, text):
        if not self.mode:
            # obtainst the term freq of every word
            tf = np.zeros(len(self.unigram))
            # We go through every word in the text
            for i in range(0, len(text)):
                if text[i].lower() in self.unigram:
                    
                    tf[self.unigram[text[i].lower()]] += 1
            tf = tf/ len(text)
        else:
            # Feature has the size of the number of distinct words received
            tf = np.zeros(len(self.bigram))
            # We go through every word in the text
            for i in range(0, len(text)-1):
                merged_words = (text[i].lower()+text[i+1].lower())
                if merged_words in self.bigram:
                    # If it contained in the unigram,
                    # It is a feature of the new text.
                    # We search for the index of the word in the text and use that index in the feature as well
                    # Then, we increase the number of occurrences of that word.
                    # Seems like the bag of words seen in class.
                    tf[self.bigram[merged_words]] += 1
            
        return tf
        
    def transform_list(self, text_set):
        # Obtains tf-idf
        vocabulary = self.unigram
        if self.mode:
            vocabulary = self.bigram


        # Obtaining the tf of every document-word
        tf = []
        for i in range(0, len(text_set)):
            tf.append(self.transform(text_set[i]))

        # Getting the document freq of every word
        df = np.zeros(len(vocabulary))
        for document in range(len(text_set)):
            for word in range(len(vocabulary)):            
                # knowing in how many documents the word appear
                if tf[document][word]>0:
                    df[word]+=1

        # Calculating the inverse document freq of a word
        idf = np.zeros(len(vocabulary))
        for word in range(len(vocabulary)):
                idf[word] = np.log(len(text_set)/(df[word]+1))

        tf_idf = []
        
        for document in range(len(text_set)):
            tmp = []
            for word in range(len(vocabulary)):
                tmp.append(tf[document][word]*idf[word])
            tf_idf.append(tmp)

        
        
        return np.array(tf_idf)


class SPARSE(FeatureExtractor):
    """customized feature extractor,BM25 (term frequency / inverse document frequency)
    """
    def __init__(self,bigram=False):
        # Add your code here!
        self.unigram = {}
        self.bigram = {}
        self.mode = bigram
        self.b = 0.75
        self.k = 1.75


    # def fit(self, text_set):
        # Creates unigrams dict
        # for i in range(len(text_set)):
        #     print(text_set[i])
        #     text_set[i] = tokenize(text_set[i])
    #     index = 0
    #     if not self.mode:
    #         for i in range(0, len(text_set)):
    #             for j in range(0, len(text_set[i])):
    #                 if text_set[i][j].lower() not in self.unigram:
    #                     self.unigram[text_set[i][j].lower()] = index
    #                     index += 1
    #                 else:
    #                     continue
    #     else:
    #         index = 0
    #         for i in range(0, len(text_set)):
    #             for j in range(0, len(text_set[i])-1):
    #                 merged_words = (text_set[i][j]+text_set[i][j+1]).lower()
    #                 if merged_words not in self.bigram:
    #                     self.bigram[merged_words] = index
    #                     index += 1
    #                 else:
    #                     continue      
    # def transform(self, text):
    #     if not self.mode:
    #         # obtainst the term freq of every word
    #         tf = np.zeros(len(self.unigram))
    #         # We go through every word in the text
    #         for i in range(0, len(text)):
    #             if text[i].lower() in self.unigram:
    #                 h = text[i].lower()
    #                 tf[self.unigram[text[i].lower()]] += 1
    #         tf = np.log10(tf + 1)
    #     else:
    #         # Feature has the size of the number of distinct words received
    #         tf = np.zeros(len(self.bigram))
    #         # We go through every word in the text
    #         for i in range(0, len(text)-1):
    #             merged_words = (text[i].lower()+text[i+1].lower())
    #             if merged_words in self.bigram:
    #                 # If it contained in the unigram,
    #                 # It is a feature of the new text.
    #                 # We search for the index of the word in the text and use that index in the feature as well
    #                 # Then, we increase the number of occurrences of that word.
    #                 # Seems like the bag of words seen in class.
    #                 tf[self.bigram[merged_words]] += 1
            
    #     return tf
        
    # def transform_list(self, text_set):
    #     # Obtains tf-idf
    #     vocabulary = self.unigram
    #     if self.mode:
    #         vocabulary = self.bigram

    #     # Obtaining the average document length
    #     avg_d = 0
    #     for text in text_set:
    #         avg_d += len(text)
    #     avg_d = avg_d/len(text_set)


    #     # Obtaining the tf of every document-word
    #     tf = []
    #     for i in range(0, len(text_set)):
    #         tf.append(self.transform(text_set[i]))

    #     # Getting the document freq of every word
    #     df = np.zeros(len(vocabulary))
    #     for document in range(len(text_set)):
    #         for word in range(len(vocabulary)):            
    #             # knowing in how many documents the word appear
    #             if tf[document][word]>0:
    #                 df[word]+=1

    #     # Calculating the inverse document freq of a word
    #     idf = np.zeros(len(vocabulary))
    #     for word in range(len(vocabulary)):
    #             idf[word] = np.log10(len(text_set)/(df[word]+1))

    #     bm25 = []
        
    #     for document in range(len(text_set)):
    #         tmp = []
    #         for word in range(len(vocabulary)):
    #             tmp.append( ( tf[document][word] / (self.k*(1-self.b + self.b * (len(text_set[document])/avg_d)) + tf[document][word] ) )  * idf[word] )
    #         bm25.append(tmp)

        
        
    #     return np.array(bm25)
    def fit(self, text_set):
        self.unigram
        index = 0
        self.corpus = [tokenize(p.lower()) for p in text_set]
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue
        self.calculate_df(self.corpus)

    def calculate_df(self,sentences):
        self.word_count = {}
        for word in self.unigram:
            self.word_count[word] = 0
            for sent in sentences:
                if word in sent:
                    self.word_count[word] += 1

    def calculate_tf(self,document, word):
        occurance = len([token for token in document if token == word])
        return np.log10(occurance + 1)
    
    def inverse_doc_freq(self,word):
        try:
            word_occurance = self.word_count[word] + 1
        except:
            word_occurance = 1
        return np.log10(len(self.corpus)/word_occurance)


    def get_scores(self, query, text_set, method='BM25'):
        text_set = [tokenize(p.lower()) for p in text_set]
        query = tokenize(query.lower())

        tf = np.zeros((len(text_set),len(query)))
        idf = np.zeros(len(query))

        # Obtaining the average document length
        avg_d = 0
        for text in text_set:
            avg_d += len(text)
        avg_d = avg_d/len(text_set)

        # calculate the score for every document
        for index in range(len(text_set)):
            # iterate through every word in the query
            for term in range(len(query)):
                tf[index][term] = self.calculate_tf(text_set[index],query[term])
        # Calculating idf vectors
        for term in range(len(query)):
            idf[term] = self.inverse_doc_freq(query[term]) 
            
        
        score = []
        
        if method=='BM25':
            for document in range(len(text_set)):
                tmp = 0
                for word in range(len(query)):
                    tmp += ( tf[document][word] / (self.k*(1-self.b + self.b * (len(text_set[document])/avg_d)) + tf[document][word] ) )  * idf[word]
                score.append(tmp)
        else:
            for document in range(len(text_set)):
                tmp = 0
                for word in range(len(query)):
                    tmp += tf[document][word] * idf[word]
                score.append(tmp)
        return score
                






def cosine_similarity(doc2, doc1):
    product = 0
    len1 = 0
    len2 = 0
    for i in range(len(doc1)):
        product += doc1[i] * doc2[i]
        len1 += math.pow(doc1[i],2)
        len2 += math.pow(doc2[i],2)
    result = product / (math.sqrt(len1)*math.sqrt(len2))
    return result
        
    




        
