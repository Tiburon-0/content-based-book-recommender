'''User-facing frontend -- describe a taste, get books back, vote on the results'''

# this is the app a reader touches, not the monitoring dashboard
# it is a pure http client of the api: no model, no .pkl, no dynamodb
#
# the feedback loop:
#   POST /retrieve  -> api runs the model, writes a row to dynamodb, returns its id
#   we hold that id in session_state
#   POST /feedback  -> api updates that same row with helpful=True/False
#   the dashboard reads those rows and computes live accuracy

import os

import requests
import streamlit as st

# the makefile passes API_URL=http://api:8000 -- resolved by container name over
# the shared network, so a changed ec2 public ip never breaks it
# the default below is for running outside docker
API_URL = os.environ.get('API_URL', 'http://127.0.0.1:8000')

# every request is bounded -- without a timeout a wedged api leaves the ui
# spinning with no error, which looks like a crash during a demo
# 90s is generous because a cold api is still pulling the model
REQUEST_TIMEOUT = 90

PLACEHOLDER_COVER = 'https://placehold.co/128x196?text=No+Cover'

st.set_page_config(page_title='Tiburon Book Recommender', page_icon='📚', layout='wide')


# --------[API CLIENT]--------

def get_health():
    '''Reads /health, or None when the api is unreachable'''

    try:
        response = requests.get(f'{API_URL}/health', timeout=10)
        return response.json()

    # nothing listening and listening-but-wedged are different failures,
    # but the user's next move is the same, so both collapse to None
    except requests.RequestException:
        return None


def post_retrieve(query, k):
    '''Asks the api for k recommendations. Returns (payload, error_message)'''

    try:
        response = requests.post(
            f'{API_URL}/retrieve',
            json={'text': query, 'k': k},
            timeout=REQUEST_TIMEOUT,
        )

        # 503 is the api's "registry pull failed" signal
        # pass its detail through verbatim -- that string names the actual cause
        if response.status_code == 503:
            return None, response.json().get('detail', 'The model is not loaded.')

        response.raise_for_status()
        return response.json(), None

    except requests.RequestException as error:
        return None, f'Could not reach the API at {API_URL} -- {type(error).__name__}: {error}'


def post_feedback(retrieval_id, helpful):
    '''Attaches a thumbs up/down to a retrieval the api already logged'''

    try:
        response = requests.post(
            f'{API_URL}/feedback',
            json={'retrieval_id': retrieval_id, 'helpful': helpful},
            timeout=30,
        )
        return response.status_code == 200

    except requests.RequestException:
        return False


# --------[SESSION STATE]--------
# streamlit reruns this whole script on every click
# anything not in session_state is destroyed the moment a feedback button is
# pressed -- losing retrieval_id there would silently break the dashboard

st.session_state.setdefault('results', None)
st.session_state.setdefault('retrieval_id', None)
st.session_state.setdefault('feedback_sent', False)


# --------[SIDEBAR: SERVICE STATUS]--------
# doubles as demo evidence -- shows the registry version actually being served

with st.sidebar:
    st.header('Service status')

    health = get_health()

    if health is None:
        st.error('API unreachable')
        st.caption(f'Tried `{API_URL}`')

    elif health.get('model_loaded'):
        st.success('Model loaded')

        served = health.get('served_by') or {}
        st.caption(f"Registry version: `{served.get('registry_version', '?')}`")
        st.caption(f"Source version: `{served.get('source_version', '?')}`")
        st.caption(f"Digest: `{str(served.get('digest', '?'))[:12]}...`")

    else:
        st.warning('API up, model not loaded')
        # /health puts the real exception in `error` -- most useful string
        # there is when a deployment misbehaves, so show it rather than hide it
        st.caption(health.get('error', 'No error reported.'))

    st.divider()
    st.caption(f'Backend: `{API_URL}`')


# --------[QUERY]--------

st.title('📚 Tiburon Book Recommender')
st.write(
    'Describe the kind of book you want -- a genre, a theme, an author you like, '
    'or a few titles you have enjoyed. The recommender matches your description '
    'against 165,744 books by text similarity.'
)

query = st.text_area(
    'What do you feel like reading?',
    placeholder='e.g. victorian detective novels with an unreliable narrator',
    height=100,
)

left, right = st.columns([1, 3])

with left:
    k = st.slider('How many books', min_value=1, max_value=20, value=8)

with right:
    st.write('')  # aligns the button with the slider
    search = st.button('Recommend', type='primary', use_container_width=True)

if search:
    if not query.strip():
        st.warning('Type something first.')

    else:
        with st.spinner('Searching the catalog...'):
            payload, error = post_retrieve(query, k)

        if error:
            st.error(error)
            st.session_state.results = None

        else:
            st.session_state.results = payload.get('recommendations', [])
            st.session_state.retrieval_id = payload.get('retrieval_id')
            st.session_state.feedback_sent = False
            st.session_state.latency_ms = payload.get('latency_ms')


# --------[RESULTS]--------

results = st.session_state.results

if results is not None:

    if not results:
        # recommend() returns [] when the query shares no vocabulary with the corpus
        # a reachable state, not an error -- explain it instead of showing a blank page
        st.info(
            'No matches. None of those words appear in the catalog vocabulary. '
            'Try describing the book in more common terms, or add an author or genre.'
        )

    else:
        st.subheader(f'{len(results)} recommendations')
        st.caption(f"Retrieved in {st.session_state.get('latency_ms', '?')} ms")

        # three across reads well on a laptop and still works on a projector
        for row_start in range(0, len(results), 3):
            columns = st.columns(3)

            # strict=False on purpose -- the last row is short whenever the result
            # count is not a multiple of 3, and zip should stop at the shorter one
            for column, book in zip(columns, results[row_start:row_start + 3], strict=False):
                with column:
                    # image is null for 24.5% of the catalog and recommend()
                    # converts those to None, so the placeholder is the common path
                    st.image(book.get('image') or PLACEHOLDER_COVER, width=128)

                    st.markdown(f"**{book.get('Title', 'Untitled')}**")
                    st.caption(book.get('authors') or 'Author unknown')
                    st.caption(book.get('categories') or 'Uncategorized')
                    st.caption(f"similarity {book.get('score', 0):.3f}")

        # --------[FEEDBACK]--------
        # this control is what makes phase 3.2's live accuracy possible

        st.divider()

        if st.session_state.retrieval_id is None:
            # log_retrieval() fails soft and returns None, so the api can still serve
            # when dynamodb is down -- no row means nothing to attach feedback to
            st.caption('This retrieval was not logged, so feedback is unavailable.')

        elif st.session_state.feedback_sent:
            st.success('Thanks -- your feedback was recorded.')

        else:
            st.write('**Were these useful?**')
            up, down, _ = st.columns([1, 1, 6])

            if up.button('👍 Yes', use_container_width=True):
                if post_feedback(st.session_state.retrieval_id, True):
                    st.session_state.feedback_sent = True
                    st.rerun()
                else:
                    st.error('Could not record feedback.')

            if down.button('👎 No', use_container_width=True):
                if post_feedback(st.session_state.retrieval_id, False):
                    st.session_state.feedback_sent = True
                    st.rerun()
                else:
                    st.error('Could not record feedback.')

        st.caption(f'retrieval_id `{st.session_state.retrieval_id}`')
