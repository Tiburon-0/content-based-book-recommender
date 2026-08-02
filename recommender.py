import joblib as jl
import numpy as np
import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer

class TiburonBookRecommender:
    '''Recommends books by text similarity based on user taste description'''

    def __init__(self, max_features=None, min_df=2, ngram_range=(1, 2)):

        self.text_fields = ('Title', 'authors', 'categories', 'description')
        self.display_fields = ('Title', 'authors', 'categories', 'image', 'ratingsCount')

        # retains solely the N most frequent terms across the corpus
        # caps memory and drops the long tail of typos and one-off words
        self.max_features = max_features

        # min_df = 2: ignores terms appearing in fewer than 2 documents
        # words used in exactly one book cannot aid comparison
        self.min_df = min_df

        # ngram_range=(1, 2): captures unigrams and bigrams
        # users might express interest in topics e.g., machine learning or artificial intelligence
        # authors might also share given names or surnames
        # two-word phrases help with precision but compound vocabulary size
        self.ngram_range = ngram_range 

        self.vectorizer = None # the fitted TFIDFVectorizer (vocabulary + IDF weights)
        self.matrix = None # sparse matrix: one row per book, one column per term
        self.catalog = None # DataFrame of display metadata, aligned to matrix rows

    # --------[FITTING]--------

    def _build_soup(self, dataframe):
        '''Concatenates each book's text fields into one document per book'''

        # filling nulls with space instead of imputation
        # missingness degrades the document gracefully instead of deleting the book or fabricating text
        soup = dataframe[self.text_fields[0]].fillna('').astype(str)

        for field in self.text_fields[1:]:

            # ' ' acts as a separator so words do not fuse between fields 
            soup = soup + ' ' + dataframe[field].fillna('').astype(str)

        return soup

    def fit(self, dataframe):
        '''Learns the vocabulary and builds the catalog's document matrix'''

        # Operationally, 'fitting' means scan every book, decide the composing vocabulary terms (TF),
        # compute each term's IDF weight, then express every book as a vector over that vocabulary. 
        # Represents an unsupervised transformation of text into numbers

        soup = self._build_soup(dataframe)

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=self.ngram_range,

            # accounts for common words (articles, conjunctions, prepositions, etc.) in the english language
            # e.g., 'the', 'and', 'of'
            stop_words='english',

            # accounts for international authors with accent marks
            strip_accents='unicode',

            # prevents dominant words (i.e., more frequently occuring words) from skewing the analysis 
            sublinear_tf=True
        )

        # creates an M x N matrix, where M represents the book titles and N represents each word comprising the title
        # .fit_transform also returns a sparse matrix of non-zeros, optimizing for memory conservation
        matrix = self.vectorizer.fit_transform(soup)

        # conversion of values from float64 to float32 
        # saving the matrix while further optimizing for memory
        self.matrix = matrix.astype(np.float32)

        # reset index during lookup so that each index in the matrix matches each index in the catalog
        self.catalog = dataframe[list(self.display_fields)].reset_index(drop=True)

        # printed diagnostics
        print(f'Vocabulary: {len(self.vectorizer.vocabulary_):,} terms')
        print(f'Matrix Dim: {self.matrix.shape[0]:,} books x {self.matrix.shape[1]:,} terms | '
              f'{self.matrix.nnz:,} non-zero terms | Size: ({self.matrix.data.nbytes / 1e6:,.1f} MB)')

        return self

    # --------[SERVING]---------


    def recommend(self, query, k=20):
        '''Returns the k books whose text is most similar to query, best first'''

        if self.vectorizer is None:
            raise RuntimeError(f'Model is not fit...Please run .fit() or .load() prior to retrieving recommendations')

        query_vector = self.vectorizer.transform([query])

        # checks for number of non-zero tokens (i.e., # of valid words)
        if query_vector.nnz == 0:
            return []

        # calculates dot-product cosine similarity score (i.e., output) vector
        # (M books x N terms) x (N terms x 1 query) = M books x 1 query
        scores = (self.matrix @ query_vector.T).toarray().ravel()

        k = max(1, min(k, scores.shape[0]))

        top = np.argsort(-scores)[:k]

        results = []

        for position in top:
            book = {
                key: (None if pd.isna(value) else value)
                for key, value in self.catalog.iloc[position].items()
            }

            book['score'] = float(scores[position])
            results.append(book)

        return results

    def save(self, path):
        '''Serializes the fitted recommender'''

        jl.dump(self, path)

    @classmethod
    def load(cls, path):
        '''Loads the saved recommender'''

        return jl.load(path)

        

        



