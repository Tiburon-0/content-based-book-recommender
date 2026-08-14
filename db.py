'''Archtecture for interactions with DynamoDB'''

import os
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

TABLE_NAME = os.environ.get('DYNAMODB_TABLE', 'book_recommender_retrievals')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

# see create_table.py -- log_date (partition) + created_at (sort)
DATE_INDEX = 'retrievals_by_date'
LOG_DATE_PREFIX = 'RETRIEVAL#'

def get_table():
    return boto3.resource('dynamodb', region_name=AWS_REGION).Table(TABLE_NAME)


def to_decimal(value):
    '''Converts a Python float to the Decimal type DynamoDB demands'''

    return Decimal(str(round(value, 6)))

def date_bucket(moment):
    '''Builds the index partition key -- one partition per UTC day'''

    return f'{LOG_DATE_PREFIX}{moment.date().isoformat()}'

def log_retrieval(query, recommendations, latency_ms, served_by, k):
    '''Writes one retrieval to DynamoDB. Returns its id, or None if the write failed'''

    retrieval_id = str(uuid.uuid4())
    logged_at = datetime.now(timezone.utc)

    trimmed = [
        {
            'title': book.get('Title'),
            'score': to_decimal(book.get('score', 0.0)),
            'categories': book.get('categories')
        }
        for book in recommendations
    ]

    item = {
        'retrieval_id': retrieval_id,
        'log_date': date_bucket(logged_at),
        'created_at': logged_at.isoformat(timespec='microseconds'),
        'query': query,
        'k': k,
        'recommendations': trimmed,
        'top_categories': [b.get('categories') for b in recommendations if b.get('categories')],
        'top_score': to_decimal(recommendations[0]['score']) if recommendations else to_decimal(0.0),
        'result_count': len(recommendations),
        'latency_ms': to_decimal(latency_ms),
        'model_version': (served_by or {}).get('source_version', 'unknown'),
        'model_digest': (served_by or {}).get('digest', 'unknown'),
        'feedback': None
    }

    try:
        get_table().put_item(Item=item)
        return retrieval_id

    # broad on purpose: missing credentials raise NoCredentialsError, not ClientError,
    # and no logging failure should ever surface as a 500 on /retrieve
    except Exception as error:
        print(f'[db] Could not log retrieval: {type(error).__name__} -- {error}')
        return None

def record_feedback(retrieval_id, helpful):
    '''Attaches a thumbs up/down to a retrieval that was already logged'''

    try:
        get_table().update_item(
            Key={'retrieval_id': retrieval_id},
            UpdateExpression='SET #f = :f, #r = :r',
            ExpressionAttributeNames={'#f': 'feedback', '#r': 'feedback_at'},
            ExpressionAttributeValues={
                ':f': bool(helpful),
                ':r': datetime.now(timezone.utc).isoformat(),
            },
            # without this, update_item upserts: an unknown id would create a phantom
            # row with no query and no latency, quietly skewing dashboard accuracy
            ConditionExpression='attribute_exists(retrieval_id)',
        )
        return True

    except Exception as error:
        print(f'[db] Could not record feedback: {type(error).__name__} -- {error}')
        return False

def fetch_recent_retrievals(limit=500, days=7):
    '''Reads logged retrievals newest-first off the date index. Returns a list of dicts'''

    table = get_table()
    today = datetime.now(timezone.utc)

    rows = []

    try:
        # one query per UTC day, walking backwards until the limit is filled
        for offset in range(days):
            if len(rows) >= limit:
                break

            response = table.query(
                IndexName=DATE_INDEX,
                KeyConditionExpression=Key('log_date').eq(date_bucket(today - timedelta(days=offset))),
                ScanIndexForward=False,   # created_at descending, so newest first
                Limit=limit - len(rows),
            )

            rows.extend(response.get('Items', []))

        return rows[:limit]

    except Exception as error:
        print(f'[db] Could not read retrievals: {type(error).__name__} -- {error}')
        return []


if __name__ == '__main__':

    print(f'Table : {TABLE_NAME}')
    print(f'Region : {AWS_REGION}')
    print()

    fake_recommendations = [
        {'Title': 'Dune', 'score': 0.41, 'categories': 'science fiction'},
        {'Title': 'Neuromancer', 'score': 0.33, 'categories': 'cyberpunk'},
    ]

    print('Writing a test retrieval')
    new_id = log_retrieval(
        query='a gripping true crime novel',
        recommendations=fake_recommendations,
        latency_ms=87.3,
        served_by={'source_version': 'v1', 'digest': 'testdigest'},
        k=2
    )
    print(f' -> retrieval_id: {new_id}')

    if new_id:
        print('Recording feedback...')
        print(f' -> {record_feedback(new_id, True)}')

        print('Reading it back...')
        rows = fetch_recent_retrievals(limit=5)
        print(f' -> {len(rows)} items(s) in the table')
        for row in rows[:3]:
            print(f'     {row.get("created_at")} | {row.get("query")!r} '
                  f'| {row.get("latency_ms")} ms | feedback={row.get("feedback")}')
    else:
        print('Write failed -- check credentials, region, and that the table exists.')
