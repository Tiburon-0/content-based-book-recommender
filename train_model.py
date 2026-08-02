import pandas as pd 
import time

from pathlib import Path

from recommender import TiburonBookRecommender

books_dataset = 'books_data.csv'

catalog_columns = ['Title', 'description', 'authors', 'categories', 'ratingsCount', 'image']

def load_and_preprocess(file, required_fields=('Title', 'authors', 'categories')):
    '''Loads dataset, cleans dataset, ands provides shape of cleaned dataset before saving'''

    print(f'Loading dataset')
    dataframe = pd.read_csv(Path(__file__).parent / file, usecols=catalog_columns)
    print(f'Dataset loaded')

    print(f'Peek at loaded data: \n {dataframe.head(10)}')
    pre_drop_size = dataframe.shape[0]
    print(f'This dataset holds {pre_drop_size} titles, each with {dataframe.shape[1]} features.')

    print(f'Assessing missing values...')
    print(f'Missing val count: \n {dataframe.isnull().sum()}')
    print(f'Missing val percentage: \n {(dataframe.isnull().sum() / len(dataframe)) * 100}')

    print(f'Cleaning data...')

    # required_fields cannot be null for sake of dataset integrity
    dataframe = dataframe.dropna(subset=list(required_fields)).reset_index(drop=True)
    post_drop_size = dataframe.shape[0]
    print(f'Dataset cleaned: {pre_drop_size - post_drop_size} titles dropped; {post_drop_size} titles remaining.')
    print(f'Peek at cleaned data:\n{dataframe.head()}')

    return dataframe

def train_model(library, max_features=None, min_df=2, ngram_range=(1, 2)):
    '''Fits the recommender's vocabulary and document matrix over the corpus'''

    recommender = TiburonBookRecommender(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range
    )

    print(f'Training the model...')

    recommender.fit(library)

    print(f'Model trained.')

    return recommender

def save_model(trained_model, file_name):
    '''Saves trained model for later access'''

    path = Path(__file__).parent / file_name

    print(f'Saving the model...')

    trained_model.save(path)

    print(f'Model saved to {file_name}: ({path.stat().st_size / 1e6:,.1f} MB).')

def smoke_test(trained_model, query='a gripping true crime novel'):
    '''Prints a set of recommendations to test model functionality and accuracy'''

    print(f'\n Commencing Smoke test -- query: {query!r}')

    for rank, book in enumerate(trained_model.recommend(query, k=5), start=1):
        print(f'{rank} | {book['Title']} | (score {book['score']:.3f})')

def run_pipeline(data, saved_file_name='tiburon_book_recommendation_model.pkl', **config):
    '''Loads and preprocesses the data, trains the model, and saves the trained model'''

    start_time = time.time()

    print(f'Running pipeline...')

    df = load_and_preprocess(data)

    recommendation_model = train_model(df, **config)

    save_model(recommendation_model, saved_file_name)

    smoke_test(recommendation_model)

    end_time = time.time()

    print(f'Pipeline Completion Time: {end_time - start_time:.2f} s')

    return f'Pipeline complete.'

if __name__ == '__main__':
    print(run_pipeline(books_dataset))



