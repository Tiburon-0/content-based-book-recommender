'''Pipeline Architecture'''

import time
from pathlib import Path

import pandas as pd

# experiment tracking
import wandb

from promote_model import REGISTRY_PATH
from recommender import TiburonBookRecommender

BOOKS_DATASET = 'books_data.csv'

CATALOG_COLUMNS = ['Title', 'description', 'authors', 'categories', 'ratingsCount', 'image']

REQUIRED_FIELDS = ('Title', 'authors', 'categories')

def load_and_preprocess(file, required_fields=REQUIRED_FIELDS):
    '''Loads dataset, cleans dataset, ands provides shape of cleaned dataset before saving'''

    print('Loading dataset')
    dataframe = pd.read_csv(Path(__file__).parent / file, usecols=CATALOG_COLUMNS)
    print('Dataset loaded')

    print(f'Peek at loaded data: \n {dataframe.head(10)}')
    pre_drop_size = dataframe.shape[0]
    print(f'This dataset holds {pre_drop_size} titles, each with {dataframe.shape[1]} features.')

    print('Assessing missing values...')
    print(f'Missing val count: \n {dataframe.isnull().sum()}')
    print(f'Missing val percentage: \n {(dataframe.isnull().sum() / len(dataframe)) * 100}')

    print('Cleaning data...')

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

    print('Training the model...')

    recommender.fit(library)

    print('Model trained.')

    return recommender

def save_model(trained_model, file_name):
    '''Saves trained model for later access'''

    path = Path(__file__).parent / file_name

    print('Saving the model...')

    trained_model.save(path)

    print(f'Model saved to {file_name}: ({path.stat().st_size / 1e6:,.1f} MB).')

    return path

def smoke_test(trained_model, query='a gripping true crime novel'):
    '''Prints a set of recommendations to test model functionality and accuracy'''

    print(f'\n Commencing Smoke test -- query: {query!r}')

    smoke_test_query_start = time.time()

    recommendations = trained_model.recommend(query, k=5)

    for rank, book in enumerate(recommendations, start=1):
        print(f'{rank} | {book['Title']} | (score {book['score']:.3f})')

    smoke_test_query_latency_ms = (time.time() - smoke_test_query_start) * 1000

    print(f'Query latency: {smoke_test_query_latency_ms:.1f} ms')

    return smoke_test_query_latency_ms

def run_pipeline(data=BOOKS_DATASET, saved_file_name='tiburon_book_recommendation_model.pkl',
                 required_fields=REQUIRED_FIELDS, **config):
    '''Loads and preprocesses the data, trains the model, and saves the trained model'''

    pipeline_start_time = time.time()

    print('Running pipeline...')

    with wandb.init(entity="tiburon_0-university-of-denver",
                    project="content-based-book-recommender",
                    config=config) as run:

        data_artifact = wandb.Artifact(name='books-data', type='dataset')
        data_artifact.add_reference((Path(__file__).parent / data).as_uri())
        run.log_artifact(data_artifact)

        df = load_and_preprocess(data, required_fields=required_fields)

        train_start = time.time()
        recommendation_model = train_model(df, **config)
        train_finish = (time.time() - train_start)

        run.config.update({
            'max_features':recommendation_model.max_features,
            'min_df':recommendation_model.min_df,
            'ngram_range':recommendation_model.ngram_range,
            'required_fields':required_fields,
            'text_fields':recommendation_model.text_fields,
            'dataset':data,
        }, allow_val_change=True)

        path = save_model(recommendation_model, saved_file_name)

        artifact = wandb.Artifact(
            name='tiburon-book-recommender',
            type='model',
            metadata={
                'catalog_size': len(df),
                'vocabulary_size': len(recommendation_model.vectorizer.vocabulary_),
                'min_df': recommendation_model.min_df,
                'ngram_range': recommendation_model.ngram_range,
            },
        )
        artifact.add_file(str(path))

        logged_artifact = run.log_artifact(artifact)

        logged_artifact.wait()

        logged_artifact.link(target_path=REGISTRY_PATH, aliases=['staging'])

        print(f'Linked to registry: {REGISTRY_PATH}:staging')

        smoke_test_query_latency = smoke_test(recommendation_model)

        pipeline_end_time = time.time()

        run.log({
            'catalog_size': len(df),
            'vocabulary_size': len(recommendation_model.vectorizer.vocabulary_),
            'matrix_nnz': recommendation_model.matrix.nnz,
            'matrix_mb': recommendation_model.matrix.data.nbytes  / 1e6,
            'artifact_mb': path.stat().st_size / 1e6,
            'train_time_s' : train_finish,
            'query_latency_ms': smoke_test_query_latency,
            'pipeline_time_s': pipeline_end_time - pipeline_start_time
        })

        print(f'Pipeline Completion Time: {pipeline_end_time - pipeline_start_time:.2f} s')

    return 'Pipeline complete.'

if __name__ == '__main__':
    print(run_pipeline(BOOKS_DATASET))

