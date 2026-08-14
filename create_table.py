'''One-time DynamoDB table creation'''

import os
import time

import boto3
from botocore.exceptions import ClientError

TABLE_NAME = os.environ.get('DYNAMODB_TABLE', 'book_recommender_retrievals')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

DATE_INDEX = 'retrievals_by_date'

# what the monitoring dashboard reads; the recommendations list is deliberately left out
INDEX_FIELDS = [
    'query', 'k', 'result_count', 'top_score', 'top_categories',
    'latency_ms', 'model_version', 'model_digest', 'feedback', 'feedback_at',
]

# every attribute named in a key schema, base table and index alike
KEY_ATTRIBUTES = [
    {'AttributeName': 'retrieval_id', 'AttributeType': 'S'},
    {'AttributeName': 'log_date', 'AttributeType': 'S'},
    {'AttributeName': 'created_at', 'AttributeType': 'S'},
]

DATE_INDEX_SCHEMA = {
    'IndexName': DATE_INDEX,

    'KeySchema': [
        {'AttributeName': 'log_date', 'KeyType': 'HASH'},
        {'AttributeName': 'created_at', 'KeyType': 'RANGE'},
    ],

    'Projection': {
        'ProjectionType': 'INCLUDE',
        'NonKeyAttributes': INDEX_FIELDS,
    },
}

def wait_for_index(table, poll_seconds=10):
    '''Blocks until the date index finishes backfilling and reports ACTIVE'''

    print(f'Waiting on {DATE_INDEX} -- backfill can take a few minutes...')

    while True:
        table.reload()

        statuses = {
            index['IndexName']: index['IndexStatus']
            for index in (table.global_secondary_indexes or [])
        }

        status = statuses.get(DATE_INDEX)

        if status is None:
            raise RuntimeError(f'{DATE_INDEX} is not attached to {TABLE_NAME}')

        if status == 'ACTIVE':
            print(f'Index {DATE_INDEX} is ACTIVE.')
            return

        print(f'  {DATE_INDEX}: {status}')

        time.sleep(poll_seconds)

def add_index(table):
    '''Attaches the date index to a table that was created without it'''

    present = {index['IndexName'] for index in (table.global_secondary_indexes or [])}

    if DATE_INDEX in present:
        print(f'Index {DATE_INDEX} is already present -- nothing to do.')
        return table

    print(f'Adding index {DATE_INDEX} to {TABLE_NAME}...')

    table.update(
        AttributeDefinitions=KEY_ATTRIBUTES,
        GlobalSecondaryIndexUpdates=[{'Create': DATE_INDEX_SCHEMA}],
    )

    wait_for_index(table)

    return table

def create_table():
    '''Creates the retrievals table and its date index if the table does not already exist'''

    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)

    try:
        table = dynamodb.create_table(
            TableName=TABLE_NAME,

            KeySchema=[
                {'AttributeName': 'retrieval_id', 'KeyType': 'HASH'}
            ],

            AttributeDefinitions=KEY_ATTRIBUTES,

            GlobalSecondaryIndexes=[DATE_INDEX_SCHEMA],

            BillingMode='PAY_PER_REQUEST'
        )

        print(f'Creating table {TABLE_NAME} in {AWS_REGION}...')

        table.wait_until_exists()

        print(f'Table {TABLE_NAME} is ACTIVE and ready.')

        wait_for_index(table)

        return table

    except ClientError as error:
        code = error.response['Error']['Code']

        if code == 'ResourceInUseException':
            print(f'Table {TABLE_NAME} already exists -- checking its indexes instead.')
            return add_index(dynamodb.Table(TABLE_NAME))

        print(f'Could not create {TABLE_NAME}: {code} -- {error}')
        raise

if __name__ == '__main__':
    create_table()
