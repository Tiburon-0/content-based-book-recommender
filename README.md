# Machine Learning Book Recommendation System Using Content-Based Filtering

## Machine Learning Pipeline

### `recommender.py`

* **Model Requirements**: This recommender system is a [content-based filtering model](https://www.geeksforgeeks.org/machine-learning/ml-content-based-recommender-system/). Though [collaborative-filtering (CF)](https://www.geeksforgeeks.org/machine-learning/collaborative-filtering-ml/) holds strong in situations in which user profiles already exist and share similar patterns in taste, the CF approach suffers from the *cold start* limitation (i.e. interactions from new users who do not have established history). By contrast, content-based filtering (CBF) is built on existing item metadata and features, making it the optimal choice for individuals who may not have established history.
  Implementation of two steps: *retrieval* and *ranking*
  * *Retrieval*: generate--retrieve--title candidates for each mentioned title
    * tests [clamping](https://www.geeksforgeeks.org/python/how-to-clamp-floating-numbers-in-python/) for *k* value
  * *Ranking*: rank those candidates, remove duplicates or consumed-titles

### `train_model.py`

* **Data Collection**: Ingests data (`Amazon Books Dataset: books_data.csv`)
* **Data Cleaning/Feature Engineering**: Dropping nulls using `load_and_preprocess()`
* **Data Labeling**: N/A - this is an unsupervised learning model
* **Feature Selection**: Loading data: only load the used features (i.e., 'Title', 'description', 'authors', 'categories', 'ratingsCount', 'image'). The remaining features (i.e., previewLink, publisher, publishedDate, and infoLink) are noise.  This helps
  streamline data parsing and cleaning, as well as storage management.
* **Model Training**: *Text frequency-inverse document frequency (TF-IDF)* - vectorizes the text; the corpus is represented as an M x N matrix. The number of titles (M) may differ from size of vocabulary (N); however, the size of the vector computed from each
  vectorized query (i.e., an 1 x N) must be compatible for matrix-vector multiplication to produce a dot product. Thus, the vector will be transposed to dimensions N x 1 to produce the dot product of M x 1, in which each title will have a score.
  Titles that are closer together in vector space will have a larger dot product or cosine similarity score. Titles are then sorted in descending order according to their scores so that similar titles are served based on the specified *k*.
* **Model Evaluation**: When building some recommendations (e.g., *You Might Like*, *Because You Viewed*, *For You*, etc.), the goal is to minimize the distance from a user's favorite title to another title, expressed as ||vt^k - vt^i||. This model uses a leave-one-out with precision@k evaluation method: consider users with 5+ liked titles, hide one title, build a query from the rest, evaluate whether the hidden book returns. Use W&B to sweep across `max_features` and `min_df` hyperparameters--strikes the balance between *memory* and *precision*.

  * Unittests:
    * `test_recommender.py`
      * `TestRecommenderFitting`
      * `TestRecommenderRetrieval`
      * `TestRecommenderSaveLoad`
    * test_api.py
      * `TestEndpointsWithoutModel`
      * `TestEndpointsWithModel`
      * `TestRequestValidation`

### Docker | Amazon Web Services

* **Model Deployment / Model Monitoring**: *Docker* containerization for environmental consistency and accessibility; *Amazon Elastic Compute Cloud* for deployment and monitoring
* NoSQL Cloud Database (DynamoDB):

  * Low-latency
  * Flexible
  * Automatic scalability
  * High Availability & Durability

### `create_table.py`

* DynamoDB Implementation:
  * For this project, one key scans several thousand items (i.e., retrieval logs). Scale would demand implementation of a composite key (partition key: `"PRED#2026-08-07",`sort key: `"2026-08-07T13:22:01Z#<uuid>")`for targeted queries. **Tradeoff**: The simple, single key is optimal for the project at hand. A composite key adds unnecessary complexity.
  * Timezone in UTC for agnostic time-tracking
  * Function as a Service (FaaS) - Only pay for what is used

### `main.py`

* Houses the fastAPI layer, which loads the Production model from the Registry
* Pydantic schemas for user TextInput validation
* Provides several endpoints:
  * / - returns the artifact
  * /health - confirms API functionality
  * /retrieve - retrieves a specified number of recommendations *k* from a single query
  * /retrieve_from_queries - retrieves a specified number of recommendations *k* from multiple queries
  * /example - provides example of retrieval functionality by returning a random title from the catalog

## [Weights and Biases](https://wandb.ai/tiburon_0-university-of-denver/projects)

### `promote_model.py`

**Linking**:

* **Supports model lifecycle management & scope promotion**:

  * Enables CLI staging and promotion of model artifacts in the *Registry* (i.e., --list, --alias production)
* **Supports data lineage**:

  * Alias acts as mutable pointer, enabling changes in production without redeploying the highest-performing artifact.
    * **Importance**: Respects possibility of **model degradation**, **concept drift**, and **data drift**
* **Provides naming consistency for artifacts**:

  * **entity**: *tiburon_0-university-of-denver*
  * **project**: *content-based-book-recommender*
  * **artifact & version**: *tiburon-book-recommender:version*
  * **Project Namespace**: *tiburon_0-university-of-denver/content-based-book-recommender/tiburon-book-recommender:v1*)

Promotional decisions are currently rooted in time and space complexity, *cost optimization* (e.g., vocabulary size, model size, latency), pending development of the evaluation harness. So, the structuring enables seamless rollback and shifts in promotion necessitated by various business needs, shifts in priority balance (speed vs. accuracy).


Project Dashboard: [wandb.ai/tiburon_0-university-of-denver/content-based-book-recommender?nw=nwusertiburon_0](https://wandb.ai/tiburon_0-university-of-denver/content-based-book-recommender?nw=nwusertiburon_0)

---

## Installation

Clone the repository: `git clone https://github.com/Tiburon-0/content-based-book-recommender`
Download [dataset](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews?select=books_data.csv)
Open Command-Line Interface:
Run `pip install -r requirements.txt`
Run `python train_model.py`

Example output:

* ![smoke test eda](assets/smoke_test_1-1.png)
* ![smoke test query output](assets/smoke_test_1-2.png)

---

## Design Considerations

* *Relevance*: A user might solely enter their favorite titles, or they might convey their taste using prose. Titles are presented in descending order based on their ranks, which are also provided.
* *Freshness*: The catalog is frozen at training time--preventing data drift--so new books would necessitate retraining. The 'publishedDate' feature is also dropped, restricting inference of recency.
* *Latency*: During the initial data cleaning process, 78% of the titles were dropped due to null values across several of the fields, reducing the dataset to 47,797 titles. After applying a more structured cleaning approach, the dataset regained a significant portion of its size, recouping 165,744 titles. Thus, the model's vocabulary grew to 1,240,874 terms, with non-zero terms growing from 2.46M to 12.7M. Serving costs ~85 ms per query, of which the sparse matrix-vector product consumes 79 ms. Thus, a 3.5x larger catalog produced an 8x slower query.
* *Diversity*: Impacted by duplicate titles with diverse casing, as exhibited in the smoke test. ('Now Wait *for* Last Year' vs. 'Now Wait *For* Last Year'). Additionally, multiple books by the same author restricts diversity. Tuning the `ngram_range` hyperparameter to (1, 2) accounts for authors with shared given or surnames (bigrams).
* *Fairness*: Some recommender systems are biased toward certain outputs (e.g., books, ads, movies, etc.) due to financial incentive. Despite there being no financial incentive to show preference toward certain titles, this model differs in that recommendations with higher rating weights are probable to be served more frequently--meaning the books with lower ratings have a lower probability of being served, keeping their ratings low. This is referred to as a *popularity feedback loop*.

---

## Learning lessons

### Framing

The underlying problem this project solves is retrieval, not prediction. I've built several classification and regression models, so this retrieval problem required a mental reframing. While the former models conventionally call `.fit` on the estimator, then `.predict`to generate the outputs, this model calls `.fit_transform` on the model, then `.transform` on any presented queries. This systematically trains the vectorizer on the corpus of titles so that it can learn each term and weight each accordingly. Once the titles and their composing terms are learned and vectorized into a higher dimension (i.e., a matrix of titles(M) x Terms(N)), any presented queries are then vectorized into that same dimension. This results in each query being transformed into a vector of 1 x N. Additionally, `.fit_transform` serves as the optimal application rather than `.fit` -> `.transform` given that `.fit_transform` enables single-pass tokenization of the corpus. However, if during the model training phase, the engineer irresponsibly calls `.fit_transform` on the query, as opposed to `.transform`, the model retrains on the query. Thus, the model's entire vocabulary becomes restricted to those words, corrupting its context of the substantial knowledgebase it learned prior.

### Evaluation

Tuning the `min_df` hyperparameter from 2 to 5 decreased the model's vocabulary by 78% and its matrix size by 18.6%. This suggests that distinct terms with higher frequencies are more prevalent across documents of 5 or more, an observation which upholds [Zipf&#39;s law](https://www.geeksforgeeks.org/nlp/zipfs-law/). Data is dominated by *common* terms; vocabulary is defined by *rarer* terms.
