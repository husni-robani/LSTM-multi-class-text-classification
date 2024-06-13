import numpy as np
import re
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from nlp_id import StopWord, Lemmatizer
from gensim.utils import tokenize
import torch
import torch.nn.functional as F

class DataSetup():
    """ A class that used for preprocessing data

    Attributes:
        X_train (np.ndarray): feature data training
        X_test (np.ndarray): feature data test
        y_train (np.ndarray): label data training
        y_test (np.ndarray): label data test
        w2v_model: pretrained word2vec model
        is_oversamping: if true, so the data train will be oversampled
    """
    def __init__(self, X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, w2v_model, is_oversampling=True) -> None:
        self.__X_train = X_train
        self.__X_test = X_test
        self.__y_train = y_train
        self.__y_test = y_test
        self.w2v_model = w2v_model
        self.__is_oversamping = is_oversampling
        
    def processing_data_train(self):
        """This Method used for preprocessing data train"""
        X_train, y_train = self.__X_train, self.__y_train

        # encode labels
        print('\n')
        with tqdm(total=(len(y_train)), desc="Encode Labels") as pbar:
            encode_fn = np.vectorize(self.__encode_study_program)
            y_train = encode_fn(y_train, pbar)
    

        # oversampling train data
        if self.__is_oversamping:
            X_train, y_train = self.__oversampling(X=X_train, y=y_train)
            X_train = X_train.squeeze()

        # remove punctuations
        print('\n')
        with tqdm(total=(len(X_train)), desc="Remove Punctuations") as pbar:
            rp_fn = np.vectorize(self.__remove_punctuations)
            X_train= rp_fn(X_train, pbar)

        # remove stopwords
        print('\n')
        with tqdm(total=(len(X_train)), desc="Remove Stopwords") as pbar:
            X_train= np.vectorize(self.__remove_stopwords)(X_train, pbar)

        # lemmatization
        print('\n')
        X_train = self.__lemmatization(X_train)

        # tokenize, vectorize, padding
        print('\n')
        X_train = self.__collate_fn(X_train)



        return X_train, y_train
        
    def processing_data_test(self):
        """This Method used for preprocessing data test"""
        X_test, y_test = self.__X_test, self.__y_test

        # encode labels
        print('\n')
        with tqdm(total=(len(y_test)), desc="Encode Labels") as pbar:
            encode_fn = np.vectorize(self.__encode_study_program)
            y_test = encode_fn(y_test, pbar)

        # remove punctuations
        print('\n')
        with tqdm(total=(len(X_test)), desc="Remove Punctuations") as pbar:
            rp_fn = np.vectorize(self.__remove_punctuations)
            X_test = rp_fn(X_test, pbar)

        # remove stopwords
        print('\n')
        with tqdm(total=(len(X_test)), desc="Remove Stopwords") as pbar:
            X_test = np.vectorize(self.__remove_stopwords)(X_test, pbar)

        # lemmatization
        print('\n')
        X_test = self.__lemmatization(X_test)

        # tokenize, vectorize, padding
        print('\n')
        X_test = self.__collate_fn(X_test)



        return X_test, y_test

    def __encode_study_program(self, label, pbar=None):
        """Encode the labels tobe numerical representation"""
        if pbar is not None:
            pbar.update(1)

        studyprogram_encoder = {
            "Akuntansi": 0, 
            "Manajemen": 1,
            "Teknik Informatika": 2,
            "Bahasa Inggris": 3,
            "Desain dan Komunikasi Visual": 4
        }

        return studyprogram_encoder[label] if label in studyprogram_encoder else  ValueError("wrong label value")

    def __oversampling(self, X, y):
        """oversampling data with ros"""
        return RandomOverSampler().fit_resample(X=X.reshape(-1, 1), y=y)
    
    def __remove_punctuations(self, text:str, pbar=None):
        """removing panctuations on text and doing case folding to be lowercase"""
        if pbar is not None:
            pbar.update(1)
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

    def __remove_stopwords(self, text:str, pbar=None):
        """removing stopwords on text"""
        if pbar is not None:
            pbar.update(1)

        stopword = StopWord()
        return stopword.remove_stopword(text)

    def __lemmatization(self, X:np.ndarray):
        """lemmatizing the text"""
        lemmatizer = Lemmatizer()
        lemmatized_texts = []

        for text in tqdm(X, desc="LEMMATIZATION", total=len(X)):
            lemmatized_text = lemmatizer.lemmatize(text)
            lemmatized_texts.append(lemmatized_text)

        return np.array(lemmatized_texts)
    
    def __tokenize(self, text:str):
        """tokenizing text to be list of word"""
        text = list(tokenize(text))
        return text
    
    def __vectorize(self, text):
        """vectorizing tokenized text to be numerical representation (128 vectors) using word2vec"""
        vectorized_text = np.array([self.w2v_model.wv.get_vector(word) for word in text if word in self.w2v_model.wv])
        return vectorized_text
    
    def __padding(self, text_vectorized):
        """padding the text if has less than 100 word and slicing it if more than 100"""
        if text_vectorized.size(1) > 100:
            text_vectorized = text_vectorized[:, :100, :]
        elif text_vectorized.size(1) < 100:
            pad_needed = 100 - text_vectorized.size(1)
            text_vectorized = F.pad(text_vectorized, (0, 0, 0, pad_needed), mode='constant', value=0)
        return text_vectorized

    def __collate_fn(self, X):
        X_temp = []
        for text in tqdm(X, desc="COLLATE FUNCTION", total=len(X)):
            # do tokenization
            text = self.__tokenize(text)

            # do vectorization & reshape to be [1, ..., ...]
            vectorized_text = self.__vectorize(text)

            # convert to tensor and unsqueeze
            vectorized_text = torch.unsqueeze(torch.from_numpy(vectorized_text), 0)

            # padding if vectorized_text.size(1) < 100 or slice to be 100 if more than 100
            vectorized_text = self.__padding(vectorized_text)
            
            # y = torch.unsqueeze(torch.tensor(y), 0)

            X_temp.append(vectorized_text)
        
        # stack the texts and labels to form a batch
        return torch.stack(X_temp, dim=0)
    
    def get_X_train(self):
        return self.__X_train
    
    def get_X_test(self):
        return self.__X_test
    
    def get_y_train(self):
        return self.__y_train
    
    def get_y_test(self):
        return self.__y_test