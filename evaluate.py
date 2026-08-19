'''Leave-one-out evaluation -- hit rate and precision@k against a popularity baseline'''

# why this exists: every metric logged so far is COST (latency, matrix size, train
# time). none of them can say one run is better than another, which is what makes
# registry promotion meaningful. this produces the quality number.
#
# the method:
#   1. find users who liked (rated 4+) several books that exist in our catalog
#   2. hide one of their books
#   3. build a query from the titles/authors/categories of the rest
#   4. ask the model for k recommendations
#   5. did the hidden book come back?
#
# run:
#   conda activate ml_engineer
#   python evaluate.py                      # 1000 users, k=10, logs to w&b
#   python evaluate.py --users 3000 --k 20
#   python evaluate.py --no-wandb           # local only, no run logged

import argparse
import random
import time

import pandas as pd

from recommender import TiburonBookRecommender

RATINGS_FILE = 'Books_rating.csv'

# a rating of 4 or 5 counts as a like
LIKED_THRESHOLD = 4

# cohort bounds
# 5 minimum because one book is too thin a description of taste to predict from
# 50 maximum to exclude degenerate accounts -- one user has 5,351 liked books,
# which is a bot, and its taste vector is meaningless
# excluding them is standard practice, and it is stated in the readme
MIN_LIKED = 5
MAX_LIKED = 50

def normalize_title(titles):
    '''Lowercases and collapses whitespace/punctuation so titles join across files'''

    # the two CSVs disagree on case and punctuation for the same book
    # e.g. 'Now Wait For Last Year' vs 'Now Wait for Last Year'
    # without this the join silently loses matches and the cohort shrinks
    return (
        titles.astype(str)
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )


def load_cohort(catalog_titles, sample_users, seed):
    '''Builds the evaluation cohort from the ratings file'''

    # usecols is mandatory -- review/text is most of the 2.86 GB
    print(f'Reading {RATINGS_FILE} ...')
    ratings = pd.read_csv(
        RATINGS_FILE,
        usecols=['Title', 'User_id', 'review/score'],
    )

    ratings = ratings.dropna(subset=['Title', 'User_id'])
    ratings = ratings[ratings['review/score'] >= LIKED_THRESHOLD]

    ratings['key'] = normalize_title(ratings['Title'])

    # keep only books the model can actually return
    ratings = ratings[ratings['key'].isin(catalog_titles)]

    liked = ratings.groupby('User_id')['key'].apply(lambda s: sorted(set(s)))
    liked = liked[liked.map(len).between(MIN_LIKED, MAX_LIKED)]

    print(f'  {len(liked):,} users with {MIN_LIKED}-{MAX_LIKED} liked in-catalog books')

    users = list(liked.index)
    random.Random(seed).shuffle(users)

    return [liked[user] for user in users[:sample_users]], ratings


def popularity_ranking(ratings, size=200):
    '''Most-reviewed titles -- the null hypothesis'''

    # popularity is famously hard to beat in recsys
    # if the model does not beat this, that is a finding worth reporting, not a
    # failure to hide
    return list(ratings['key'].value_counts().head(size).index)


def build_query(catalog, key_to_row, keys):
    '''Turns a user's liked books into the text they would plausibly have typed'''

    # catalog holds Title, authors, categories (display_fields) -- not description
    # that is fine and arguably more realistic: a real user types titles and authors
    # they like, not full jacket copy
    parts = []

    for key in keys:
        row = catalog.iloc[key_to_row[key]]
        parts.append(' '.join(
            str(row[field]) for field in ('Title', 'authors', 'categories')
            if pd.notna(row[field])
        ))

    return ' '.join(parts)


def score_run(hits, users, k):
    '''Hit rate and precision@k from a hit count'''

    # only ONE book is held out per user, so at most one recommendation can be
    # relevant -- that caps precision@k at 1/k mechanically
    # hit rate is the interpretable number here; precision is reported because the
    # rubric names it
    hit_rate = hits / users if users else 0.0

    return {'hit_rate': hit_rate, 'precision': hit_rate / k}


def evaluate(model, cohort, catalog, key_to_row, popular, k):
    '''Runs both arms over the cohort and returns their scores'''

    model_hits = 0
    popularity_hits = 0
    latencies = []

    for index, liked in enumerate(cohort, start=1):

        if index % 250 == 0:
            print(f'  {index}/{len(cohort)} users')

        # hold out the last book, query from the rest
        hidden, shown = liked[-1], liked[:-1]
        query = build_query(catalog, key_to_row, shown)

        # ask for extra candidates so that after removing books the user already
        # liked there are still k left -- recommending a book they have read is
        # not a hit, it is a leak
        start = time.time()
        raw = model.recommend(query, k=k + len(shown))
        latencies.append((time.time() - start) * 1000)

        seen = set(shown)
        recommended = [
            key for key in normalize_title(pd.Series([b['Title'] for b in raw]))
            if key not in seen
        ][:k]

        if hidden in recommended:
            model_hits += 1

        # same rule for the baseline
        baseline = [key for key in popular if key not in seen][:k]

        if hidden in baseline:
            popularity_hits += 1

    users = len(cohort)

    return {
        'model': score_run(model_hits, users, k),
        'popularity': score_run(popularity_hits, users, k),
        'mean_query_latency_ms': sum(latencies) / len(latencies) if latencies else 0.0,
        'users_evaluated': users,
        'k': k,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='tiburon_book_recommendation_model.pkl')
    parser.add_argument('--users', type=int, default=1000)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    print(f'Loading {args.model} ...')
    model = TiburonBookRecommender.load(args.model)

    catalog = model.catalog
    catalog_keys = normalize_title(catalog['Title'])

    # first occurrence wins on duplicate titles -- the catalog is not unique by title
    key_to_row = {}
    for row, key in enumerate(catalog_keys):
        key_to_row.setdefault(key, row)

    cohort, ratings = load_cohort(set(catalog_keys), args.users, args.seed)
    popular = popularity_ranking(ratings)

    print(f'Evaluating {len(cohort)} users at k={args.k} ...')
    results = evaluate(model, cohort, catalog, key_to_row, popular, args.k)

    lift = (
        results['model']['hit_rate'] / results['popularity']['hit_rate']
        if results['popularity']['hit_rate'] else float('inf')
    )

    print()
    print(f"Users evaluated      : {results['users_evaluated']:,}")
    print(f"k                    : {results['k']}")
    print(f"Model hit rate       : {results['model']['hit_rate']:.4f}")
    print(f"Model precision@k    : {results['model']['precision']:.4f}")
    print(f"Popularity hit rate  : {results['popularity']['hit_rate']:.4f}")
    print(f"Popularity precision : {results['popularity']['precision']:.4f}")
    print(f"Lift over popularity : {lift:.2f}x")
    print(f"Mean query latency   : {results['mean_query_latency_ms']:.1f} ms")

    if args.no_wandb:
        return

    # logged as a separate run type so evaluation runs are distinguishable from
    # training runs in the w&b ui
    import wandb

    with wandb.init(
        entity='tiburon_0-university-of-denver',
        project='content-based-book-recommender',
        job_type='evaluation',
        config={'k': args.k, 'users': args.users, 'seed': args.seed,
                'min_liked': MIN_LIKED, 'max_liked': MAX_LIKED},
    ) as run:
        run.log({
            'hit_rate_at_k': results['model']['hit_rate'],
            'precision_at_k': results['model']['precision'],
            'popularity_hit_rate_at_k': results['popularity']['hit_rate'],
            'popularity_precision_at_k': results['popularity']['precision'],
            'lift_over_popularity': lift,
            'mean_query_latency_ms': results['mean_query_latency_ms'],
            'users_evaluated': results['users_evaluated'],
        })

    print('\nLogged to W&B.')


if __name__ == '__main__':
    main()
