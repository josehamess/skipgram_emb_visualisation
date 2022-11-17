import regex as re
import pickle
from collections import Counter


class Cleaner():
    def __init__(self, stop_words):
        self.stop_words = stop_words
    

    def formatting_cleaner(self, text):

        # removes format tokens from text #
        # returns the cleaned text #

        text = re.sub(r'\[[0-9]"\]', '', text)
        text = re.sub(r'<br>', '', text)
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\t', '', text)
        text = re.sub(r'\d', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text
    

    def decapitalise(self, text):

        # removes capital letters from text #
        # returns clean list of text #

        decap_dict = {  'A':'a', 'B':'b', 'C':'c', 'D':'d', 
                        'E':'e', 'F':'f', 'G':'g', 'H':'h', 
                        'I':'i', 'J':'j', 'K':'k', 'L':'l', 
                        'M':'m', 'N':'n', 'O':'o', 'P':'p', 
                        'Q':'q', 'R':'r', 'S':'s', 'T':'t',
                        'U':'u', 'V':'v', 'W':'w', 'X':'x', 
                        'Y':'y', 'Z':'z'
                        }
        decap_text = ''
        for i in range(len(text)):
            if re.match(r'[A-Z]', text[i]):
                decap_text += decap_dict[text[i]]
            else:
                decap_text += text[i]
        
        return decap_text
    

    def punctuation_removal(self, text):

        # removes punctuation from text #
        # returns new list of text #

        clean_text = ''
        for i in range(len(text)):
            if text[i] not in [ ',', 
                                '(', ')', '*', "'", '-', 
                                ':', '/', '£', ';', '[', ']', '"']:
                clean_text += text[i]
        clean_text = re.sub(r'\s+', ' ', clean_text)
    
        return clean_text
    

    def sentence_splitter(self, text):

        # splits text into sentences #
        # return list of sentences #

        return re.split(r'[.?!]', text)
    

    def stop_word_removal(self, sentences):

        # splits sentences into words and removes stop words #
        # returns list of clean sentences #

        cleaned_sentences = []
        ngram_counts = {}
        for sentence in sentences:
            split_sentence = sentence.split(' ')
            cleaned_sentence = []
            for ngram in split_sentence:
                if ngram not in self.stop_words and len(ngram) > 0:
                    cleaned_sentence.append(ngram)
                    if ngram in ngram_counts:
                        ngram_counts[ngram] += 1
                    else:
                        ngram_counts[ngram] = 1
            cleaned_sentences.append(cleaned_sentence)

        return cleaned_sentences, ngram_counts
    

    def remove_sentence_punctuation(self, text):

        # remove puctuation that hasnt been removed #
        # return cleaned text #

        clean_text = ''
        for char in text:
            if char not in ['.', '?', '!']:
                clean_text += char
        
        return clean_text
    

    def clean(self, text, save):

        # cleans text #
        # returns cleaned sentences #

        print(f'... cleaning text ...')
        print('')
        text = self.punctuation_removal(self.decapitalise(self.formatting_cleaner(text)))
        print(f'... splitting into sentences ...')
        sentences, ngram_counts = self.stop_word_removal(self.sentence_splitter(text))
        print('')
        print(f'number of sentences extracted: {len(sentences)}')

        if save == True:
            with open('data/sentences', 'wb') as fp:
                pickle.dump(sentences, fp)

        return sentences, ngram_counts