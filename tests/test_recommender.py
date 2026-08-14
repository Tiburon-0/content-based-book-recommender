'''Unit tests for the recommender -- no network, no files, no aws'''

# a recommender fails silently: a misaligned catalog still returns k books with
# plausible scores and never raises
# these tests are the cheap defence against that

import pandas as pd

from recommender import TiburonBookRecommender


class TestBuildSoup:
    '''The text-joining step, where nulls do their damage'''

    def test_nulls_do_not_poison_the_whole_string(self, sample_books):
        # in pandas 'text' + NaN == NaN
        # book 4 has a null description, so unfilled nulls would make its ENTIRE
        # soup NaN -- still in the catalog, still recommendable, matching nothing
        model = TiburonBookRecommender(min_df=1)
        soup = model._build_soup(sample_books)

        assert soup.notna().all(), 'a null field turned an entire document into NaN'
        assert 'Machine Learning' in soup.iloc[3]

    def test_soup_contains_every_text_field(self, sample_books):
        model = TiburonBookRecommender(min_df=1)
        soup = model._build_soup(sample_books)

        first = soup.iloc[0]
        assert 'Victorian Detective' in first     # Title
        assert 'Arthur Doyle' in first            # authors
        assert 'Detective mystery' in first       # categories
        assert 'foggy london' in first            # description

    def test_fields_are_separated_by_a_space(self, sample_books):
        # without the ' ' separator the last word of one field fuses to the first
        # word of the next, inventing tokens like 'DetectiveArthur'
        model = TiburonBookRecommender(min_df=1)
        soup = model._build_soup(sample_books)

        assert 'DetectiveArthur' not in soup.iloc[0]


class TestFitting:
    '''Fitting must leave the matrix and the catalog in lockstep'''

    def test_matrix_and_catalog_have_the_same_number_of_rows(self, fitted_model, sample_books):
        # the alignment invariant
        # recommend() reads catalog.iloc[i] for matrix row i, so any difference in
        # length or order returns the wrong book with a plausible score attached
        assert fitted_model.matrix.shape[0] == len(fitted_model.catalog)
        assert len(fitted_model.catalog) == len(sample_books)

    def test_catalog_index_is_reset(self, sample_books):
        # a filtered DataFrame keeps its original index (3, 7, 11...)
        # iloc is positional so it still works, but any later .loc reads the wrong row
        filtered = sample_books[sample_books['ratingsCount'].notna()]
        model = TiburonBookRecommender(min_df=1).fit(filtered)

        assert list(model.catalog.index) == list(range(len(filtered)))

    def test_matrix_is_float32(self, fitted_model):
        # halves the matrix -- 50.8 MB instead of ~102 MB on the real catalog
        # matters on a 1 GB instance
        assert fitted_model.matrix.dtype == 'float32'

    def test_min_df_drops_terms_seen_only_once(self, sample_books):
        # 'frontier' appears in exactly one book
        # tested explicitly because the shared fixture uses min_df=1
        loose = TiburonBookRecommender(min_df=1).fit(sample_books)
        strict = TiburonBookRecommender(min_df=2).fit(sample_books)

        assert 'frontier' in loose.vectorizer.vocabulary_
        assert 'frontier' not in strict.vectorizer.vocabulary_


class TestRecommend:
    '''Retrieval behaviour, including the cases that return nothing'''

    def test_returns_the_requested_number_of_books(self, fitted_model):
        assert len(fitted_model.recommend('detective mystery', k=3)) == 3

    def test_results_are_sorted_best_first(self, fitted_model):
        results = fitted_model.recommend('machine learning', k=5)
        scores = [book['score'] for book in results]

        assert scores == sorted(scores, reverse=True)

    def test_finds_the_obviously_relevant_book(self, fitted_model):
        # sanity check, not a metric -- if a query this literal misses,
        # something structural is broken
        titles = [b['Title'] for b in fitted_model.recommend('machine learning', k=2)]

        assert any('Machine Learning' in title for title in titles)

    def test_unknown_words_return_an_empty_list(self, fitted_model):
        # a query sharing no vocabulary produces a zero vector
        # scoring it would return k books at score 0.0 -- arbitrary results that
        # look real, so [] is what lets the frontend say "no matches" honestly
        assert fitted_model.recommend('zzzz qqqq xxxx', k=5) == []

    def test_k_larger_than_the_catalog_is_clamped(self, fitted_model):
        # asking for 500 from a 6-book catalog must not raise
        assert len(fitted_model.recommend('detective', k=500)) == 6

    def test_every_result_carries_the_display_fields(self, fitted_model):
        book = fitted_model.recommend('detective', k=1)[0]

        for field in ('Title', 'authors', 'categories', 'image', 'ratingsCount', 'score'):
            assert field in book

    def test_nulls_come_back_as_none_not_nan(self, fitted_model):
        # NaN is not valid json -- fastapi would fail to serialize it
        # 24.5% of real images are null, so this is the common path
        results = fitted_model.recommend('victorian baking cakes', k=6)
        images = [book['image'] for book in results]

        assert any(image is None for image in images)
        assert not any(isinstance(image, float) for image in images)

    def test_score_is_a_plain_float(self, fitted_model):
        # numpy.float32 is not json-serializable either
        score = fitted_model.recommend('detective', k=1)[0]['score']

        assert type(score) is float


class TestSaveLoad:
    '''joblib round-trip -- how the api gets a ready-to-serve object in one call'''

    def test_loaded_model_returns_identical_results(self, fitted_model, tmp_path):
        path = tmp_path / 'model.pkl'
        fitted_model.save(path)

        reloaded = TiburonBookRecommender.load(path)

        before = fitted_model.recommend('detective mystery', k=3)
        after = reloaded.recommend('detective mystery', k=3)

        assert [b['Title'] for b in before] == [b['Title'] for b in after]

    def test_loaded_model_keeps_its_vocabulary_and_catalog(self, fitted_model, tmp_path):
        path = tmp_path / 'model.pkl'
        fitted_model.save(path)

        reloaded = TiburonBookRecommender.load(path)

        assert reloaded.vectorizer.vocabulary_ == fitted_model.vectorizer.vocabulary_
        pd.testing.assert_frame_equal(reloaded.catalog, fitted_model.catalog)


class TestUnfittedModel:

    def test_recommending_before_fitting_raises(self):
        # guards against serving a model that was constructed but never fitted
        # otherwise it fails deep inside sklearn with a confusing message
        model = TiburonBookRecommender()

        try:
            model.recommend('anything')
            raise AssertionError('expected a RuntimeError')
        except RuntimeError as error:
            assert 'not fit' in str(error).lower()
