from typeguard import typechecked
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import spacy
from typing import Union
from sklearn.model_selection import train_test_split



class DataPreProcessing:
    @typechecked
    def __init__(self, df:pd.DataFrame):
        self.df = df

    def split(self, test_percent: float):
        train, test = train_test_split(df, test_size = test_percent, random_state = 7)

    def tfidf(self, train_data: pd.Series, split_data: list = None):
        tfidf_vectorizer = TfidfVectorizer()
        bow_train = tfidf_vectorizer.fit_transform(train_data)
        if split_data:
            for i in range(len(split_data)):
                split_data[i] = tfidf_vectorizer.transform(split_data[i])
            split_data.insert(0, bow_train)    
            return split_data
        else:
            return bow_train



    

class TextPreProcessing:
    @typechecked
    def __init__(self, sentences: Union[str,pd.Series]):
        self.sentences = sentences 

    def flatten(self, lst: list):
        new_lst = []
        self.flatten_helper(lst, new_lst)
        return new_lst
 
    def flatten_helper(self, lst: list, new_lst: list):
        for element in lst:
            if isinstance(element, list):
                self.flatten_helper(element, new_lst)
            else:
                new_lst.append(element)    

    def remove_small_words(self, Series):
        t = Series.str.split(expand=True).stack()
        return t.loc[t.str.len() >= 4].groupby(level=0).apply(' '.join)

    def preprocessing(self):
        a_lemmas = []
        nlp = spacy.load('en_core_web_md')
        #Streaming Pre-Processing
        if type(self.sentences) == str:
            corpus = nlp(self.sentences) 
            lemmas = [token.lemma_ for token in corpus]
            a_lemmas.append(pd.Series([lemma for lemma in lemmas if (lemma.isalpha() and nlp.vocab[lemma].is_stop==False)]))
            a_lemmas = self.remove_small_words(a_lemmas[0]) 
            a_lemmas.reset_index(inplace = True, drop = True)
            a_lemmas = ' '.join(a_lemmas)
            return a_lemmas
        else:
            #Batch Pre-Processing
            self.sentences = self.sentences.fillna('-EMPTY-')
            corpus = list(nlp.pipe(self.sentences))
            for i in range(len(corpus)):
                try:
                    #creating name entity recognition list for the especific corpus
                    ents = [ent.text.split() for ent in corpus[i].ents]
                    ents = self.flatten(ents)    
                    # Tokenization with lemmatizer
                    lemmas = [token.lemma_ for token in corpus[i]]
                    # Removing non-alphabetic characters
                    a_lemmas.append(pd.Series([lemma for lemma in lemmas if (lemma.isalpha() and nlp.vocab[lemma].is_stop==False)]))
                    a_lemmas[i] = self.remove_small_words(a_lemmas[i])
                    a_lemmas[i].reset_index(inplace = True, drop = True)
                except:
                    a_lemmas[i] = '-EMPTY-'
                a_lemmas[i] = ' '.join(a_lemmas[i])
            return a_lemmas



            


                

