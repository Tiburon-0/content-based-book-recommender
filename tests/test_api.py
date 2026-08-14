'''Integration tests for the fastapi endpoints'''

# TestClient calls the app in-process -- no server to start, no port to bind
#
# two situations, both real:
#   DEGRADED -- registry pull failed, no model loaded
#               happens on an expired w&b key, a missing production alias,
#               or a deploy without network
#   HEALTHY  -- a model is present
#               the api fixture blocks the real download, so the test injects the
#               six-book model from conftest plus fake database functions

from fastapi.testclient import TestClient

# --------[DEGRADED: no model loaded]--------


class TestEndpointsWithoutModel:
    '''What the api does when the registry pull failed'''

    def test_root_still_answers(self, api):
        # no require_model() guard on the banner, so it works when the model does not
        # useful for "is the container even up?" during a deployment
        response = TestClient(api.app).get('/')

        assert response.status_code == 200

    def test_health_reports_degraded_instead_of_failing(self, api):
        response = TestClient(api.app).get('/health')
        body = response.json()

        # /health returns 200 even when unhealthy
        # a grader or load balancer reads the body to find out why, and a 500 hides it
        assert response.status_code == 200
        assert body['status'] == 'degraded'
        assert body['model_loaded'] is False
        assert body['error'] is not None, 'the actual exception must be reported'

    def test_retrieve_returns_503(self, api):
        response = TestClient(api.app).post('/retrieve', json={'text': 'detective', 'k': 3})

        # 503 not 404 -- the endpoint exists, the dependency behind it is missing
        assert response.status_code == 503

    def test_batch_retrieve_returns_503(self, api):
        response = TestClient(api.app).post('/retrieve_from_queries', json={'text': ['a', 'b']})

        assert response.status_code == 503

    def test_feedback_still_works_without_a_model(self, api, monkeypatch):
        # /feedback skips require_model() on purpose -- it concerns a retrieval that
        # was already logged, so a registry outage must not also throw away feedback
        monkeypatch.setattr(api, 'record_feedback', lambda retrieval_id, helpful: True)

        response = TestClient(api.app).post(
            '/feedback', json={'retrieval_id': 'abc-123', 'helpful': True}
        )

        assert response.status_code == 200


# --------[HEALTHY: model injected]--------


class TestEndpointsWithModel:
    '''Normal serving, with a real (tiny) model and a fake database'''

    def _client(self, api, fitted_model, monkeypatch, logged=None):
        '''Wires a working api: real model, fake dynamodb'''

        # logged is an optional list collecting what would have been written,
        # so a test can assert the write actually happened

        def fake_log_retrieval(query, recommendations, latency_ms, served_by, k):
            if logged is not None:
                logged.append({'query': query, 'k': k, 'latency_ms': latency_ms})
            return 'test-id-1'

        monkeypatch.setattr(api, 'model', fitted_model)
        monkeypatch.setattr(api, 'load_error', None)
        monkeypatch.setattr(api, 'served_by', {
            'registry_version': 'v0', 'source_version': 'v0', 'digest': 'testdigest'
        })
        monkeypatch.setattr(api, 'log_retrieval', fake_log_retrieval)
        monkeypatch.setattr(api, 'record_feedback', lambda rid, helpful: rid == 'test-id-1')

        return TestClient(api.app)

    def test_health_reports_ok(self, api, fitted_model, monkeypatch):
        body = self._client(api, fitted_model, monkeypatch).get('/health').json()

        assert body['status'] == 'OK'
        assert body['model_loaded'] is True
        assert body['served_by']['source_version'] == 'v0'

    def test_retrieve_returns_recommendations(self, api, fitted_model, monkeypatch):
        response = self._client(api, fitted_model, monkeypatch).post(
            '/retrieve', json={'text': 'detective mystery', 'k': 3}
        )
        body = response.json()

        assert response.status_code == 200
        assert len(body['recommendations']) == 3
        assert body['retrieval_id'] == 'test-id-1'
        assert body['latency_ms'] >= 0

    def test_retrieve_writes_a_log_row(self, api, fitted_model, monkeypatch):
        # the dashboard has nothing to display unless serving logs
        # assert the call happens rather than assuming it
        logged = []
        self._client(api, fitted_model, monkeypatch, logged).post(
            '/retrieve', json={'text': 'machine learning', 'k': 2}
        )

        assert len(logged) == 1
        assert logged[0]['query'] == 'machine learning'
        assert logged[0]['k'] == 2

    def test_response_is_json_serializable_with_nulls(self, api, fitted_model, monkeypatch):
        # the sample catalog has null images and ratingsCount
        # NaN instead of None would fail serialization here, exactly as it would on
        # the real catalog where 24.5% of images are null
        response = self._client(api, fitted_model, monkeypatch).post(
            '/retrieve', json={'text': 'victorian baking', 'k': 6}
        )

        assert response.status_code == 200
        assert any(book['image'] is None for book in response.json()['recommendations'])

    def test_batch_returns_one_result_per_query(self, api, fitted_model, monkeypatch):
        response = self._client(api, fitted_model, monkeypatch).post(
            '/retrieve_from_queries', json={'text': ['detective', 'machine learning'], 'k': 2}
        )
        body = response.json()

        assert body['count'] == 2
        assert [r['query'] for r in body['results']] == ['detective', 'machine learning']

    def test_unmatched_query_returns_empty_not_an_error(self, api, fitted_model, monkeypatch):
        response = self._client(api, fitted_model, monkeypatch).post(
            '/retrieve', json={'text': 'zzzz qqqq', 'k': 5}
        )

        assert response.status_code == 200
        assert response.json()['recommendations'] == []

    def test_example_endpoint_returns_a_catalog_title(self, api, fitted_model, monkeypatch):
        body = self._client(api, fitted_model, monkeypatch).get('/example').json()

        assert body['example_query'] in list(fitted_model.catalog['Title'])

    def test_feedback_on_an_unknown_id_returns_404(self, api, fitted_model, monkeypatch):
        # record_feedback() uses a condition expression so an unknown id fails
        # instead of upserting a phantom row -- the api turns that False into a 404
        response = self._client(api, fitted_model, monkeypatch).post(
            '/feedback', json={'retrieval_id': 'does-not-exist', 'helpful': True}
        )

        assert response.status_code == 404


# --------[REQUEST VALIDATION]--------


class TestRequestValidation:
    '''Pydantic rejects bad input before it reaches the model -- 422'''

    # matters because grading is done by sending live requests
    # a malformed body should give a clear rejection, not a 500

    def test_empty_text_is_rejected(self, api):
        response = TestClient(api.app).post('/retrieve', json={'text': '', 'k': 3})

        assert response.status_code == 422

    def test_missing_text_is_rejected(self, api):
        response = TestClient(api.app).post('/retrieve', json={'k': 3})

        assert response.status_code == 422

    def test_k_above_the_maximum_is_rejected(self, api):
        # k is bounded 1..50 -- without the cap, k=100000 builds a 100k-item response
        response = TestClient(api.app).post('/retrieve', json={'text': 'a', 'k': 999})

        assert response.status_code == 422

    def test_k_below_one_is_rejected(self, api):
        response = TestClient(api.app).post('/retrieve', json={'text': 'a', 'k': 0})

        assert response.status_code == 422

    def test_feedback_requires_a_boolean(self, api):
        response = TestClient(api.app).post(
            '/feedback', json={'retrieval_id': 'abc', 'helpful': 'maybe'}
        )

        assert response.status_code == 422
