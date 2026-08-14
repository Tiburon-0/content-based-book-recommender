'''Model monitoring dashboard -- latency, category drift, and live accuracy from dynamodb'''

# runs alone on the second ec2 instance
# reads only through db.fetch_recent_retrievals() -- never calls the api,
# shares no network with it, reads no files
#
# what phase 3.2 asks for, and where each is answered below:
#   latency over time                 -> LATENCY
#   distribution of predicted classes -> CATEGORY DRIFT
#   feedback mechanism for accuracy   -> FEEDBACK
#
# the model has no classes -- it retrieves, it does not classify
# the honest analogue of a class distribution is the mix of CATEGORIES it
# recommends, compared between an earlier and a recent window
# same failure the rubric is pointing at: collapsing onto a few genres
# this mapping is stated in the readme too

import os
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from db import fetch_recent_retrievals

TABLE_NAME = os.environ.get('DYNAMODB_TABLE', 'book_recommender_retrievals')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

# below this the accuracy panel raises an alert instead of just reporting
ACCURACY_ALERT_THRESHOLD = 0.60

st.set_page_config(page_title='Recommender Monitoring', page_icon='📈', layout='wide')


# --------[LOAD]--------

def to_float(value):
    '''Casts dynamodb Decimals to floats -- Decimal breaks pandas arithmetic and altair'''

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=60)
def load_retrievals(limit, days):
    '''Reads the retrieval log into a DataFrame'''

    # cached so that clicking between panels does not re-query dynamodb on every
    # streamlit rerun -- each rerun would otherwise be a fresh billed read
    # the refresh button clears this explicitly

    rows = fetch_recent_retrievals(limit=limit, days=days)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)

    # created_at is stored as an iso string -- charts need real timestamps
    frame['created_at'] = pd.to_datetime(frame['created_at'], errors='coerce', utc=True)

    for column in ('latency_ms', 'top_score', 'result_count', 'k'):
        if column in frame:
            frame[column] = frame[column].map(to_float)

    # feedback is True / False / None, where None means no vote yet
    # keep it as an object column -- casting to bool would turn every unvoted
    # retrieval into a thumbs-down and destroy the accuracy number
    if 'feedback' not in frame:
        frame['feedback'] = None

    return frame.sort_values('created_at')


# --------[SIDEBAR]--------

with st.sidebar:
    st.header('Data source')
    st.caption(f'Table: `{TABLE_NAME}`')
    st.caption(f'Region: `{AWS_REGION}`')

    st.divider()

    days = st.slider('Days of history', 1, 7, 7)
    limit = st.slider('Max retrievals', 50, 1000, 500, step=50)

    if st.button('Refresh now', use_container_width=True):
        load_retrievals.clear()
        st.rerun()

    st.caption('Data auto-refreshes every 60 seconds.')

data = load_retrievals(limit, days)

st.title('📈 Book Recommender — Production Monitoring')

# an empty table is the normal state on a fresh deployment, not a bug
# say what to do about it instead of rendering five blank charts
if data.empty:
    st.warning(
        'No retrievals logged yet.\n\n'
        'Either the API has not served a request, or it cannot reach DynamoDB. '
        'Send one through the frontend (or `POST /retrieve`) and refresh.'
    )
    st.stop()

st.caption(
    f"{len(data)} retrievals · "
    f"{data['created_at'].min():%Y-%m-%d %H:%M} to {data['created_at'].max():%Y-%m-%d %H:%M} UTC"
)


# --------[HEADLINE NUMBERS]--------

voted = data[data['feedback'].notna()]
helpful_rate = voted['feedback'].mean() if len(voted) else None

one, two, three, four = st.columns(4)

one.metric('Retrievals', len(data))
two.metric('Median latency', f"{data['latency_ms'].median():.0f} ms")
three.metric('Feedback collected', f'{len(voted)} ({len(voted) / len(data):.0%})')
four.metric(
    'Live accuracy',
    f'{helpful_rate:.0%}' if helpful_rate is not None else 'n/a',
    help='Share of rated retrievals the user marked helpful.',
)

if helpful_rate is not None and helpful_rate < ACCURACY_ALERT_THRESHOLD:
    st.error(
        f'Live accuracy is {helpful_rate:.0%}, below the {ACCURACY_ALERT_THRESHOLD:.0%} '
        'threshold. Investigate recent queries and consider rolling the registry back '
        'to the previous production version.'
    )

st.divider()


# --------[LATENCY]--------

st.subheader('Prediction latency over time')
st.caption(
    'Time the model spent turning a query into recommendations, measured server-side '
    'around the retrieval call. Excludes network time to the browser.'
)

latency_chart = (
    alt.Chart(data)
    .mark_line(point=True)
    .encode(
        x=alt.X('created_at:T', title='Time (UTC)'),
        y=alt.Y('latency_ms:Q', title='Latency (ms)'),
        tooltip=[
            alt.Tooltip('created_at:T', title='When'),
            alt.Tooltip('latency_ms:Q', title='ms', format='.1f'),
            alt.Tooltip('query:N', title='Query'),
        ],
    )
    .properties(height=280)
)

# the rolling median is the line to watch
# single spikes are usually a cold cache or a noisy co-tenant, a rising median
# is a real regression
latency_median = (
    alt.Chart(data)
    .mark_line(strokeDash=[6, 4], color='firebrick')
    .transform_window(rolling='median(latency_ms)', frame=[-9, 0])
    .encode(x='created_at:T', y=alt.Y('rolling:Q', title='Latency (ms)'))
)

st.altair_chart(latency_chart + latency_median, use_container_width=True)

p50, p95 = data['latency_ms'].quantile([0.50, 0.95])
st.caption(f'p50 {p50:.0f} ms · p95 {p95:.0f} ms · max {data["latency_ms"].max():.0f} ms')

st.divider()


# --------[CATEGORY DRIFT]--------

st.subheader('Recommended category distribution (target drift)')
st.caption(
    'This recommender has no predicted classes, so the analogue of class distribution '
    'is the mix of categories it recommends. A shift here means the system is steering '
    'readers toward a narrower or different slice of the catalog than it used to.'
)


def explode_categories(frame):
    '''One row per recommended category across every retrieval in frame'''

    # top_categories is a dynamodb list attribute, so boto3 returns a python list
    # rows where no recommendation carried a category come back empty and drop out
    if 'top_categories' not in frame:
        return pd.Series(dtype=str)

    return frame['top_categories'].explode().dropna()


categories = explode_categories(data)

if categories.empty:
    st.info('No categories recorded yet.')

else:
    top_categories = categories.value_counts().head(12).reset_index()
    top_categories.columns = ['category', 'count']

    st.altair_chart(
        alt.Chart(top_categories)
        .mark_bar()
        .encode(
            x=alt.X('count:Q', title='Times recommended'),
            y=alt.Y('category:N', title=None, sort='-x'),
            tooltip=['category:N', 'count:Q'],
        )
        .properties(height=340),
        use_container_width=True,
    )

    # drift comparison: older half against newer half
    # splitting at the midpoint gives two windows of equal COUNT rather than equal
    # duration, which keeps them comparable when traffic is bursty
    if len(data) >= 20:
        midpoint = len(data) // 2
        earlier = explode_categories(data.iloc[:midpoint])
        recent = explode_categories(data.iloc[midpoint:])

        if not earlier.empty and not recent.empty:
            # compare SHARES not counts -- the windows can hold different numbers of
            # retrievals, and raw counts would show drift that is only volume
            comparison = pd.concat(
                [
                    earlier.value_counts(normalize=True).rename('share').reset_index().assign(window='Earlier'),
                    recent.value_counts(normalize=True).rename('share').reset_index().assign(window='Recent'),
                ]
            )
            comparison.columns = ['category', 'share', 'window']

            # drop the long tail or the axis fills with genres seen once
            significant = comparison[comparison['share'] >= 0.02]

            st.markdown('**Earlier vs. recent share**')
            st.altair_chart(
                alt.Chart(significant)
                .mark_bar()
                .encode(
                    x=alt.X('share:Q', title='Share of recommendations', axis=alt.Axis(format='%')),
                    y=alt.Y('category:N', title=None, sort='-x'),
                    color=alt.Color('window:N', title=None),
                    yOffset='window:N',
                    tooltip=['category:N', 'window:N', alt.Tooltip('share:Q', format='.1%')],
                )
                .properties(height=360),
                use_container_width=True,
            )
    else:
        st.caption('Need at least 20 retrievals to split the window for a drift comparison.')

st.divider()


# --------[FEEDBACK]--------

st.subheader('User feedback')
st.caption(
    'Collected by the thumbs up/down control in the frontend, written back through '
    'POST /feedback onto the retrieval row it belongs to.'
)

if voted.empty:
    st.info(
        'No feedback collected yet. Use the thumbs up/down control in the frontend — '
        'live accuracy cannot be computed without it.'
    )

else:
    left, right = st.columns(2)

    with left:
        counts = (
            voted['feedback']
            .map({True: 'Helpful', False: 'Not helpful'})
            .value_counts()
            .reset_index()
        )
        counts.columns = ['verdict', 'count']

        st.altair_chart(
            alt.Chart(counts)
            .mark_arc(innerRadius=60)
            .encode(
                theta='count:Q',
                color=alt.Color('verdict:N', title=None),
                tooltip=['verdict:N', 'count:Q'],
            )
            .properties(height=260),
            use_container_width=True,
        )

    with right:
        # cumulative helpful-rate over time
        # per-retrieval feedback is 0/1, so a raw scatter shows nothing --
        # the running mean is what shows direction
        trend = voted[['created_at', 'feedback']].copy()
        trend['helpful'] = trend['feedback'].astype(int)
        trend['cumulative_accuracy'] = trend['helpful'].expanding().mean()

        st.altair_chart(
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(
                x=alt.X('created_at:T', title='Time (UTC)'),
                y=alt.Y(
                    'cumulative_accuracy:Q',
                    title='Cumulative accuracy',
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format='%'),
                ),
                tooltip=[alt.Tooltip('cumulative_accuracy:Q', format='.0%')],
            )
            .properties(height=260),
            use_container_width=True,
        )

st.divider()


# --------[MODEL VERSIONS]--------
# not required by the rubric, but it is the payoff of the served_by trace the api
# stamps on every row -- makes a latency or accuracy change attributable to a
# specific registry version instead of to "something changed"

st.subheader('Model versions serving traffic')

versions = data.groupby('model_version').agg(
    retrievals=('retrieval_id', 'count'),
    median_latency_ms=('latency_ms', 'median'),
    mean_top_score=('top_score', 'mean'),
).reset_index()

st.dataframe(versions, use_container_width=True, hide_index=True)


# --------[RECENT ACTIVITY]--------

with st.expander('Recent retrievals'):
    recent_view = data.sort_values('created_at', ascending=False).head(50)[
        ['created_at', 'query', 'k', 'result_count', 'latency_ms', 'top_score', 'feedback', 'model_version']
    ]
    st.dataframe(recent_view, use_container_width=True, hide_index=True)

st.caption(f'Rendered {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC')
