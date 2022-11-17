import numpy as np
import matplotlib.pyplot as plt


class Helper():
    def __init__(self, w_size, emb_size, num_neg_samples, total_sentences, min_count):
        self.w_size = w_size
        self.emb_size = emb_size
        self.num_neg_samples = num_neg_samples
        self.total_sentences = total_sentences
        self.min_count = min_count
    

    def ngram_counter(self, sentences):

        # counts ngrams in setences #
        # returns dictionary of ngram counts #

        ngram_count = {}
        for sentence in sentences:
            for ngram in sentence:
                if ngram in ngram_count:
                    ngram_count[ngram] += 1
                else:
                    ngram_count[ngram] = 1
        
        return ngram_count


    def get_num_samples(self, sentences):

        # determines the size of the sample array #
        # returns an integer #

        num_samples = 0
        for sentence in sentences:
            num_samples += len(sentence)

        return num_samples 
    

    def crop_sentences(self, sentences):
        
        # removes sentences that are too short and non abundent words #
        # returns cleaned sentences #

        sentences_subset = sentences[0:self.total_sentences]
        ngram_count = self.ngram_counter(sentences_subset)
        sentences_cleaned = []
        for sentence in sentences_subset:
            new_sentence = []
            for ngram in sentence:
                if ngram_count[ngram] > self.min_count:
                    new_sentence.append(ngram)
            if len(new_sentence) > 1:
                sentences_cleaned.append(new_sentence)

        return sentences_cleaned
    

    def place_ones(self, vocab, ngram, context_ngrams):

        # creates datapoint #
        # returns a 1d binary array #
        x_vals = np.zeros((len(vocab)))
        y_vals = np.zeros((len(vocab)))
        x_inds = [x for x, y in enumerate(vocab) if y == ngram]
        y_inds = [x for x, y in enumerate(vocab) if y in context_ngrams]
        x_vals[x_inds] = 1
        y_vals[y_inds] = 1
        return np.append(x_vals, y_vals, axis=0)


    def get_samples(self, sentences):
        
        # creates a dictionary of samples for building embeddings #
        # returns a dictionary of samples #

        # clean sentences
        sentences = self.crop_sentences(sentences)
        ngram_count = self.ngram_counter(sentences)
        vocab = list(ngram_count.keys())
        print(f'num sentences: {len(sentences)}')
        print(f'vocab size: {len(vocab)}')
        
        # create samples
        samples = np.zeros((self.get_num_samples(sentences), 2 * len(vocab)))
        sample_count = 0
        for sentence in sentences:
            for i, ngram in enumerate(sentence):
                left_w = max([0, i - self.w_size])
                right_w = min([len(sentence), i + self.w_size + 1])
                context_ngrams = sentence[left_w:right_w]
                context_ngrams = [x for x in context_ngrams if x != ngram]
                samples[sample_count, :] = self.place_ones(vocab, ngram, context_ngrams)
                sample_count += 1
        
        print(f'num samples: {samples.shape[0]}')

        return samples, ngram_count
    

    def emb_dot_prod(self, emb_1, emb_2):

        # performs dot prod on embs induvidually then sums #
        # returns a vector of values #

        dot_prod = np.zeros((emb_1.shape[0]))
        for j in range(emb_1.shape[1]):
            dot_prod += emb_1[:, j] * emb_2[:, j]
        
        return dot_prod
    

    def cosine_sim(self, emb_1, emb_2):

        # calculates cosine #
        # returns cosine of two vecs #

        mod_1 = (np.sqrt(np.sum(np.square(emb_1))))
        mod_2 = (np.sqrt(np.sum(np.square(emb_2))))

        return np.dot(emb_1, emb_2) / (mod_1 * mod_2)
    

    def get_cosine_dist(self, main_embeddings_df, vocab):

        # returns a dictionary of cosine sims for vocab #

        cosine_dist_dict = {}
        for ngram_1 in vocab:
            cosine_dist_dict[ngram_1] = {}
            ngram_1_emb = main_embeddings_df.loc[ngram_1].values
            for ngram_2 in vocab:
                ngram_2_emb = main_embeddings_df.loc[ngram_2].values
                cosine_dist_dict[ngram_1][ngram_2] = 1 - self.cosine_sim(ngram_1_emb, ngram_2_emb)
        return cosine_dist_dict
    

    def get_unit_vector(self, vector):

        # turns a vector into a unit vector #
        # returns the unit vector #

        return vector / np.sqrt(np.square(vector[1, 0]) + np.square(vector[1, 1]))
    

    def plot_embs(self, main_embeddings_df, cosine_dist_dict, num_embs, ngram_choice):

        # returns a plot of the vector space of all words #

        plt.figure(figsize=(10, 10))
        ngram_cosine_dists = cosine_dist_dict[ngram_choice]
        sorted_ngrams = {k: v for k, v in sorted(ngram_cosine_dists.items(), 
                        key=lambda item: item[1])}
        sorted_ngrams.pop(ngram_choice)
        close_ngrams = list(sorted_ngrams.keys())[0:num_embs + 1]
        far_ngrams = list(sorted_ngrams.keys())[-num_embs:]
        emb = np.append(np.zeros((1, 2)), 
                        np.reshape(main_embeddings_df.loc[ngram_choice].values, 
                        (1, self.emb_size)), 
                        0)
        emb = self.get_unit_vector(emb)
        plt.plot(emb[:, 0], emb[:, 1], c='b', label=ngram_choice)
        for ngram in close_ngrams:
            emb = np.append(np.zeros((1, 2)), 
                            np.reshape(main_embeddings_df.loc[ngram].values, 
                            (1, self.emb_size)), 
                            0)
            emb = self.get_unit_vector(emb)
            plt.plot(emb[:, 0], emb[:, 1], c='g', label=ngram)
        for ngram in far_ngrams:
            emb = np.append(np.zeros((1, 2)), 
                    np.reshape(main_embeddings_df.loc[ngram].values, 
                    (1, self.emb_size)), 
                    0)
            emb = self.get_unit_vector(emb)
            plt.plot(emb[:, 0], emb[:, 1], c='r', label=ngram)
        plt.xlabel('feature 1')
        plt.ylabel('feature 2')
        plt.title('Vector representations of words')
        plt.grid()
        plt.legend()
        plt.show()