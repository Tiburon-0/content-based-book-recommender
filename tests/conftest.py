'''Shared pytest fixtures'''

# ci runs on a clean ubuntu box -- no CSVs, no aws credentials, no w&b key
# so nothing here reads a file, calls aws, or calls w&b
# the catalog is typed out by hand and the database is faked

import sys

import pandas as pd
import pytest

from recommender import TiburonBookRecommender


@pytest.fixture
def sample_books():
    '''Six-book catalog, small enough to reason about by hand'''

    # includes a null description, image, author and ratingsCount because all four
    # are common in the real data
    # "Victorian Baking" sits next to "The Victorian Detective" on purpose -- that
    # pair is the failure the smoke test found

    return pd.DataFrame({
        'Title': [
            'The Victorian Detective',
            'Victorian Baking',
            'Deep Space Nine',
            'A Guide to Machine Learning',
            'Machine Learning in Practice',
            'The Silent Witness',
        ],
        'authors': [
            "['Arthur Doyle']",
            "['Mary Berry']",
            None,
            "['Andrew Ng']",
            "['Andrew Ng']",
            "['Arthur Doyle']",
        ],
        'categories': [
            "['Detective mystery']",
            "['Cooking']",
            "['Science fiction']",
            "['Computers']",
            "['Computers']",
            "['Detective mystery']",
        ],
        'description': [
            'A victorian detective solves a murder mystery in foggy london',
            'Classic victorian cakes and baking recipes from london kitchens',
            'A space station on the frontier of known space',
            None,
            'A practical guide to machine learning models and training',
            'A detective mystery with an unreliable narrator',
        ],
        'image': [
            'http://example.com/1.jpg',
            None,
            'http://example.com/3.jpg',
            'http://example.com/4.jpg',
            None,
            'http://example.com/6.jpg',
        ],
        'ratingsCount': [12.0, None, 5.0, None, None, 3.0],
    })


@pytest.fixture
def fitted_model(sample_books):
    '''Recommender fitted on the six sample books'''

    # min_df=1 here, not the production default of 2
    # with six documents min_df=2 would throw away most of the vocabulary
    # the production setting gets its own test instead

    model = TiburonBookRecommender(min_df=1, ngram_range=(1, 1))
    model.fit(sample_books)
    return model


@pytest.fixture
def api(monkeypatch):
    '''Imports main.py with the w&b download blocked, and returns the module'''

    # main.py calls wandb.Api().artifact(...) at IMPORT time, not inside a function
    # so a plain `import main` in a test would try to download the real 173 MB model
    # -- slow locally, and a hang or credentials failure in ci
    #
    # making wandb.Api raise sends main.py down the failure path it already handles:
    # model stays None, load_error is filled in, protected endpoints answer 503
    #
    # sys.modules.pop so each test re-runs main.py fresh -- python caches modules,
    # so without it a model injected by one test would leak into the next

    import wandb

    def blocked(*args, **kwargs):
        raise RuntimeError('W&B is blocked during tests')

    monkeypatch.setattr(wandb, 'Api', blocked)
    sys.modules.pop('main', None)

    import main

    yield main

    sys.modules.pop('main', None)
