import torch
import numpy as np
from tqdm.auto import tqdm
from gensim.utils import tokenize
import re
import pandas as pd
from nlp_id.lemmatizer import Lemmatizer
from nlp_id.stopword import StopWord 
import torch.nn.functional as F



class Preprocessor():
    def __init__(self):
        self.model = None

    def study_program_encoder(self, df):
        studyprogram_encoder = {
            "Akuntansi": 0, 
            "Manajemen": 1,
            "Teknik Informatika": 2,
            "Bahasa Inggris": 3,
            "Desain dan Komunikasi Visual": 4
        }

        return df['study_program'].map(studyprogram_encoder)
    
    # function for cleaning text
    def text_cleaning(self, text):
        result = text.lower() # apply lowercas
        result = result.replace('-', ' ') # get rid of punctuations
        result = result.replace('+', ' ')
        result = result.replace('..', ' ')
        result = result.replace('.', ' ')
        result = result.replace(',', ' ')
        result = result.replace('\n', ' ') # get rid of new line
        result = re.findall('[a-z\s]', result, flags=re.UNICODE) # only use text character (a-z) and space
        result = "".join(result)
        final = " ".join(result.split())
        
        return final    # clean text
    
    # function for removing stop words
    def remove_stopwords(self, text):
        stopword = StopWord()
        return stopword.remove_stopword(text)

    
    def tokenize(self, text):
        text = list(tokenize(text))
        return text

    def vectorize(self, text):
        # vectorize word to vector with word2vec model
        vectorized_text = np.array([self.model.wv.get_vector(word) for word in text if word in self.model.wv])
        return vectorized_text
    
    def padding(self, text_vectorized):
        if text_vectorized.size(1) > 100:
            text_vectorized = text_vectorized[:, :100, :]
        elif text_vectorized.size(1) < 100:
            pad_needed = 100 - text_vectorized.size(1)
            text_vectorized = F.pad(text_vectorized, (0, 0, 0, pad_needed), mode='constant', value=0)
        return text_vectorized

    def collate_fn(self, batch):
        batch_texts = []
        batch_labels = []
        for X, y in batch:
            # do tokenization
            text = self.tokenize(X)

            # do vectorization & reshape to be [1, ..., ...]
            vectorized_text = self.vectorize(text)

            # convert to tensor and unsqueeze
            vectorized_text = torch.unsqueeze(torch.from_numpy(vectorized_text), 0)

            # padding if vectorized_text.size(1) < 100 or slice to be 100 if more than 100
            vectorized_text = self.padding(vectorized_text)
            
            # y = torch.unsqueeze(torch.tensor(y), 0)

            batch_texts.append(vectorized_text)
            batch_labels.append(y)
        
        # stack the texts and labels to form a batch
        batch_texts = torch.stack(batch_texts, dim=0)
        batch_labels = torch.tensor(batch_labels)
        
        return batch_texts, batch_labels
    
    def lemmatization(self, texts, batch_size=100):
        lemmatizer = Lemmatizer()
        batch_results = []
        for i in range(0, len(texts), batch_size):
            # Extract the current batch of texts
            batch = texts[i:i + batch_size]

            # Lemmatize the current batch
            lemmatized_texts = [lemmatizer.lemmatize(text) for text in batch]
            batch_results.extend(lemmatized_texts)

        return np.array(batch_results)