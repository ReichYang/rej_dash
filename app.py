#!/usr/bin/env python
# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
import re
from datetime import datetime as dt
from dateutil.parser import parse
from pytz import timezone
import bs4
import datetime
import numpy as np
import pandas as pd
import textblob
import lexicalrichness
import textstat
import dash_bio as dashbio
import dash_table
from dash import no_update
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
from jupyter_dash import JupyterDash
import plotly.express as px
import dash
import plotly.graph_objects as go
import plotly.tools as tls
import itertools
from tick.plot import plot_point_process
from tick.hawkes import (SimuHawkes, HawkesKernelTimeFunc, HawkesKernelExp,
                         HawkesEM, SimuHawkesSumExpKernels, HawkesSumExpKern, HawkesExpKern)
from collections import Counter
from nltk.collocations import *
import nltk
import mailbox
import email.utils

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# In[2]:

   # In[3]:

mbox = mailbox.mbox('Rej.mbox')
mbox2 = mailbox.mbox('rej2.mbox')
     # # Email Processing

     # In[4]:


def get_html_text(html):
            try:
                return bs4.BeautifulSoup(html, "html5lib").body.get_text(' ', strip=True)
            except AttributeError:  # message contents empty
                return None


class GmailMboxMessage():
            def __init__(self, email_data):
                if not isinstance(email_data, mailbox.mboxMessage):
                    raise TypeError(
                        'Variable must be type mailbox.mboxMessage')
                self.email_data = email_data
                self.labels = self.date = self.efrom = self.eto = self.subject = self.text = None

            def parse_email(self):
                email_labels = self.email_data['X-Gmail-Labels']
                email_date = self.email_data['Date']
                email_from = self.email_data['From']
                email_to = self.email_data['To']
                email_subject = self.email_data['Subject']
                email_text = self.read_email_payload()

                self.labels = email_labels
                self.date = email_date
                self.efrom = email_from
                self.eto = email_to
                self.subject = email_subject
                self.text = email_text

            def read_email_payload(self):
                email_payload = self.email_data.get_payload()
                if self.email_data.is_multipart():
                    email_messages = list(
                        self._get_email_messages(email_payload))
                else:
                    email_messages = [email_payload]
                return [self._read_email_text(msg) for msg in email_messages]

            def _get_email_messages(self, email_payload):
                for msg in email_payload:
                    if isinstance(msg, (list, tuple)):
                        for submsg in self._get_email_messages(msg):
                            yield submsg
                    elif msg.is_multipart():
                        for submsg in self._get_email_messages(msg.get_payload()):
                            yield submsg
                    else:
                        yield msg

            def _read_email_text(self, msg):
                content_type = 'NA' if isinstance(
                    msg, str) else msg.get_content_type()
                encoding = 'NA' if isinstance(msg, str) else msg.get(
                    'Content-Transfer-Encoding', 'NA')
                if 'text/plain' in content_type and 'base64' not in encoding:
                    msg_text = msg.get_payload()
                elif 'text/html' in content_type and 'base64' not in encoding:
                    msg_text = get_html_text(msg.get_payload())
                elif content_type == 'NA':
                    msg_text = get_html_text(msg)
                else:
                    msg_text = None
                return (content_type, encoding, msg_text)

        # In[5]:
emails = []
num_entries = len(mbox)
for idx, email_obj in enumerate(mbox):
    email_data = GmailMboxMessage(email_obj)
    email_data.parse_email()
    emails.append(email_data)
        #     print('Parsing email {0} of {1}'.format(idx, num_entries))

        # In[6]:

num_entries = len(mbox2)
for idx, email_obj in enumerate(mbox2):
    email_data = GmailMboxMessage(email_obj)
    email_data.parse_email()
    emails.append(email_data)
        #     print('Parsing email {0} of {1}'.format(idx, num_entries))

        # In[7]:

        # construct the dataframe
email_df = pd.DataFrame()
for e in emails:
            email_df = email_df.append(
                [{'date': e.date, 'from': e.efrom, 'to': e.eto, 'subject': e.subject, 'text': e.text}])

        # In[8]:

        # In[9]:

        # In[10]:

email_df['date_n'] = pd.to_datetime(email_df.date)

        # In[11]:

email_df['date_es'] = email_df['date_n'].apply(
            lambda x: x.astimezone(timezone('US/Eastern')))

        # In[ ]:

        # In[ ]:

        # In[12]:

email_df['weekdays'] = email_df.date_es.apply(
            lambda x: dt.strftime(x, "%A"))

        # In[13]:

email_df['hour'] = email_df.date_es.apply(
            lambda x: dt.strftime(x, "%I %p")
        )

        # In[ ]:

        # In[14]:

day_list = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        # In[ ]:

        # # Text Mining

        # In[15]:


# nltk.download('stopwords')

stop = list(stopwords.words('english'))
stop.extend(['yukun', 'yukun yang', 'yang', 'data', 'scientist'])


def extract_text(text_list):

            tags = [component[0] for component in text_list]

            real_content = None
            if 'text/html' in tags:
                ind = [component[0]
                       for component in text_list].index('text/html')
                real_content = text_list[ind][-1]
                if real_content == 'None':
                    real_content = None

            elif 'text/plain' in tags:
                ind = [component[0]
                       for component in text_list].index('text/plain')
                real_content = text_list[ind][-1]
            elif 'NA' in tags:
                ind = [component[0] for component in text_list].index('NA')
                real_content = text_list[ind][-1]

            if (real_content is not None):
                if len(real_content) > 10000:
                    real_content = None

            return real_content

def clean_text(text):

            if text is not None:

                text = re.sub('=\n', '', text)

                text = re.sub('\S*@\S*\s?', '',  text)

                text = ' '.join(word.strip(string.punctuation)
                                for word in text.split())

                text = re.sub(
                    r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '', text, flags=re.MULTILINE)

            #     text=re.sub(r'\..*\..* ?', '', text, flags=re.MULTILINE)

                text = re.sub(r"\d+", "", text)

                text = re.sub(r"={1}.{2}", "", text)

                text = text.replace('size', '').replace(
                    'text size', '').replace('adjust', '').replace('td', '')

                return text
            else:
                return None


# In[16]:

email_df['extracted'] = email_df.text.apply(extract_text)
email_df['cleaned'] = email_df.extracted.apply(clean_text)

import gensim
import spacy

def sent_to_words(sentences):
            for sentence in sentences:
                yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):

            new_docs = []
            for doc in texts:
                new_docs.append([word for word in doc if word not in stop])
            return new_docs

def make_bigrams(texts):
            return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
            return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
            """https://spacy.io/api/annotation"""
            texts_out = []
            for sent in texts:
                doc = nlp(" ".join(sent))
                texts_out.append(
                    [token.lemma_ for token in doc if token.pos_ in allowed_postags])
            return texts_out

data = email_df[email_df.cleaned.notna()].cleaned.values.tolist()

        # In[27]:

data_words = list(sent_to_words(data))

        # In[28]:

# higher threshold fewer phrases.
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

        # In[29]:

        # Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

        # Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

        # In[30]:

try:
            nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
except:
            import en_core_web_sm
            nlp = en_core_web_sm.load()

        # Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=[
                                        'NOUN', 'ADJ', 'VERB', 'ADV'])

        # print(data_lemmatized[:1])

        # In[31]:

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
        # Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
texts = data_lemmatized

        # Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
            """
            Compute c_v coherence for various number of topics

            Parameters:
            ----------
            dictionary : Gensim dictionary
            corpus : Gensim corpus
            texts : List of input texts
            limit : Max num of topics

            Returns:
            -------
            model_list : List of LDA topic models
            coherence_values : Coherence values corresponding to the LDA model with respective number of topics
            """
            coherence_values = []
            model_list = []
            for num_topics in range(start, limit, step):
                model = gensim.models.ldamodel.LdaModel(
                    corpus=corpus, num_topics=num_topics, id2word=id2word, random_state=2020)
                model_list.append(model)
                coherencemodel = CoherenceModel(
                    model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence_values.append(coherencemodel.get_coherence())

            return model_list, coherence_values


# In[77]:
model_list, coherence_values = compute_coherence_values(
            dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=2)

        # In[78]:

x = range(2, 40, 2)

        # In[79]:

choose_k = pd.DataFrame(
            {'# of Topics': x, 'coherence': coherence_values})

def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
            # Init output
            sent_topics_df = pd.DataFrame()

            # Get main topic in each document
            for i, row_list in enumerate(ldamodel[corpus]):
                row = row_list[0] if ldamodel.per_word_topics else row_list
                # print(row)
                row = sorted(row, key=lambda x: (x[1]), reverse=True)
                # Get the Dominant topic, Perc Contribution and Keywords for each document
                for j, (topic_num, prop_topic) in enumerate(row):
                    if j == 0:  # => dominant topic
                        wp = ldamodel.show_topic(topic_num)
                        topic_keywords = ", ".join([word for word, prop in wp])
                        sent_topics_df = sent_topics_df.append(pd.Series(
                            [int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                    else:
                        break
            sent_topics_df.columns = ['Dominant_Topic',
                                      'Perc_Contribution', 'Topic_Keywords']

            # Add original text to the end of the output
            contents = pd.Series(texts)
            sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
            return(sent_topics_df)


def important_words(metric, ranks):
    #     stop = list(stopwords.words('english'))
    #     stop.extend(['yukun','yukun yang','yang','data','scientist'])

    if metric == 'tf':
        vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words=stop)
        vectors = vectorizer.fit_transform(email_df.dropna().cleaned.to_list())
    elif metric == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=stop)
#         tf_vectorizer= CountVectorizer(ngram_range=(1,3),stop_words=stop)
        vectors = vectorizer.fit_transform(email_df.dropna().cleaned.to_list())

    # making df
    rankings = pd.DataFrame(vectors.todense().tolist(), columns=vectorizer.get_feature_names(
    )).sum().reset_index().rename(columns={'index': 'word', 0: 'value'})

#     if rankings.value.dtype !='int':
    rankings['value'] = rankings['value'].round(2)
    # making distinguish
    rankings['type'] = None
    for ind, row in rankings.iterrows():
        num = len(row['word'].split())
        if num == 1:
            rankings.loc[ind, 'type'] = 'unigram'
        elif num == 2:
            rankings.loc[ind, 'type'] = 'bigram'
        elif num == 3:
            rankings.loc[ind, 'type'] = 'trigram'

    return rankings.sort_values('value', ascending=False).head(ranks)


def make_important_graphs(df):

    bar = px.bar(df, y='value', x='word', text='value', color='type',
                 template='seaborn', title='Important Terms Bar Chart')
    bar.update_layout(xaxis_categoryorder='total descending')

    tree = px.treemap(df, path=['type', 'word'], values='value',
                      template='seaborn', title='Important Terms Tree Map')
    return bar, tree


# In[53]:


def collo(metric):

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    # trigram_measures = nltk.collocations.TrigramAssocMeasures()

    text = ' '.join(i for i in email_df.dropna().cleaned.to_list())
    words = [word for word in text.lower().split() if word not in stop]

    # change this to read in your data
    finder = BigramCollocationFinder.from_words(words)

    # only bigrams that appear 3+ times
    finder.apply_freq_filter(3)

    # return the 10 n-grams with the highest PMI
    # finder.nbest(bigram_measures.pmi, 15)
#     finder.nbest(bigram_measures.likelihood_ratio, 15)
    if metric == 'pmi':
        coli = finder.score_ngrams(bigram_measures.pmi)
    elif metric == 'chisquare':
        coli = finder.score_ngrams(bigram_measures.chi_sq)
    elif metric == 'likelihood_ratio':
        coli = finder.score_ngrams(bigram_measures.likelihood_ratio)

    return coli


def make_circos(test_co):

    colors = ['#fff1d6', '#c9a03d', '#02b1a0', '#848484', '#cfcfcf', '#a2e8eb', '#ecd1fc', "#f0b3c5", "#c4e5d6", "#d7f2fd", '#feb408',
              '#f09654', '#ee6f37', '#6da393', '#007890', '#e8c8ee', '#ffa3a3', '#f6e777']

    colors.extend(colors)

    top20 = test_co[:10]
    workcounter = Counter()
    for i in top20:
        workcounter.update(i[0])
    ideogram = []
    for tu in workcounter.most_common():
        ideogram.append({'id': tu[0], 'label': tu[0],
                         'color': colors.pop(), 'len': tu[-1]})

    ribbons = []
    for co in top20:
        ribbons.append({'color': '#9ecaf6',
                        'source': {'id': co[0][0], 'start': 0, 'end': 1},
                        'target': {'id': co[0][-1], 'start': 0, 'end': 1}})

    return dashbio.Circos(
        id='circos',
        layout=ideogram,
        size=600,
        selectEvent={'hover': "hover", 'click': "click", 2: "both"},
        #                 eventDatum={"0": "hover", "1": "click", "2": "both"},
        config={'ticks': {'display': False}, 'labels': {
           'size': 10, 'position': 'center'}},
        tracks=[{
            'type': 'CHORDS',
                    'data': ribbons,
            'selectEvent': {"0": "hover", "1": "click", "2": "both"},
            'tooltipContent': {
                'source': 'source',
                'sourceID': 'id',
                'target': 'target',
                'targetID': 'id',
                'targetEnd': 'end'
            }
        }
        ]
    )


# In[25]:


# View
# print(corpus[:1])


# In[32]:


# In[ ]:


# In[40]:



# df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_lemmatized)

# # Format
# df_dominant_topic = df_topic_sents_keywords.reset_index()
# df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
# df_dominant_topic.head(10)


# In[ ]:


# # Define all functions for the app

# In[ ]:



# In[81]:


def intro():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H3("Rejection Email Analytics"),
            #             html.H3("Welcome to the Clinical Analytics Dashboard"),
            html.Blockquote(
                id="intro",
                children=["Rejection hurts. Yes but yet, I've met nobody who has not been rejected.\
                It is a part of life and instead of drowning in the sorrow and somberness of being rejected, we can make it fun and try to try to analyze it.",
                          ]
            ),
            html.Div(children=['👋 My name is ',
                               html.A("Yukun", href='#contact_info'),
                               ", I graduated in this crazy time of the year and have collected 100 rejection emails from all kinds of employers during these 3 months.\
            Here I am applying my Data Science skills in Interactive Data Viz, Temporal Point Process Modelling, and Text Mining to analyze these emails. Hope you enjoy it!😀"]),
            html.Br(),
            html.H4('Instructions'),
            html.Div(children=[
                html.P("This dashboard enables multiple ways for you to interact with the plots. Every plot can be zoomed and selected, along with hovering tooltips. Despite these basics, it also supports the following interactions:"),
                html.Li(
                    'Subsetting the dataset by selecting the time range, the weekdays, and the hours.'),
                html.Li('Showing specific data entries by clicking on the Heatmap.'),
                html.Div('Other interaction options are detailed in the corresponding part')])
            #                 html.Li(
            #                     'Change the Solver of the Temporal Process and the number of days to prdict.'),
            #                 html.Li('Change the metric to rank the important terms, and the number of topics to model.')])
        ],
    )


# In[55]:


def control_card():
    return html.Div(
        id="control-card",
        children=[
            html.Strong("Select Date Range"),
            dcc.DatePickerRange(
                id='my-date-picker-range',
                min_date_allowed=email_df['date_es'].min().date(),
                max_date_allowed=email_df['date_es'].max().date(),
                initial_visible_month=dt(2020, 4, 5),
                start_date=email_df['date_es'].min().date(),
                end_date=email_df['date_es'].max().date()
            ),
            html.Br(),
            html.Br(),
            html.Strong("Select the Day of a Week"),
            dcc.Dropdown(
                id='weekdays',
                options=[{'label': day, 'value': day} for day in day_list],
                value=day_list,
                multi=True
                #                 style=dict(
                # #                     height='100%',
                #                     display='block',
                #                     verticalAlign="middle"
                #                 )
            ),
            html.Br(),
            html.Br(),
            html.Strong("Select Specific Hours"),
            dcc.Checklist(
                id='time',
                options=[{'label': t, 'value': t}
                         for t in[datetime.time(i).strftime("%I %p") for i in range(24)]],
                value=[datetime.time(i).strftime("%I %p") for i in range(24)],
                labelStyle={'display': 'inline-block'}
            ),
        ],
    )


# In[56]:


def filter_df(start, end, weekdays, time):
    filtered_df = email_df.sort_values("date_es").set_index("date_es")[
        start.astimezone(timezone('US/Eastern')):end.astimezone(timezone('US/Eastern'))
    ]

    filtered_df = filtered_df[filtered_df.weekdays.isin(weekdays)]
    filtered_df = filtered_df[filtered_df.hour.isin(time)]
    return filtered_df.reset_index()


# In[57]:


def generate_paco(filtered):

    filtered.loc[filtered.date_es.dt.weekday.isin([6, 5]), 'Is Weekend'] = True
    filtered['Is Weekend'] = filtered['Is Weekend'].fillna(False)
    filtered['Day Time'] = ((filtered.date_es.dt.hour % 24 + 4) // 4).map({1: 'Late Night',
                                                                           2: 'Early Morning',
                                                                           3: 'Morning',
                                                                           4: 'Noon',
                                                                           5: 'Evening',
                                                                           6: 'Night'})

    groupedby = filtered.groupby(['Is Weekend', 'weekdays', 'Day Time', 'hour'])[
        'from'].count().reset_index()

    new_df = pd.merge(left=filtered, right=groupedby, left_on=[
                      'Is Weekend', 'weekdays', 'Day Time', 'hour'], right_on=['Is Weekend', 'weekdays', 'Day Time', 'hour'])

    fig = px.parallel_categories(data_frame=new_df,
                                 dimensions=['Is Weekend',
                                             'weekdays', 'Day Time', 'hour'],
                                 color='from_y',
                                 labels={'weekdays': 'Day in the Week',
                                         'hour': 'Hour in the Day'},
                                 color_continuous_scale=px.colors.sequential.dense)

    fig.layout['coloraxis']['colorbar']['title']['text'] = 'Count'
    fig.update_layout({'height': 600})

    fig.layout.margin = {'t': 30, 'l': 10, 'r': 10, 'b': 20}
    return fig


# In[84]:


# f=generate_paco(email_df)
# f.layout.margin=['t':30, 'l':10, 'r':10]


# In[59]:


def generate_patient_volume_heatmap(start, end, hm_click, reset, weekdays, time):
    """
    :param: start: start date from selection.
    :param: end: end date from selection.
    :param: clinic: clinic from selection.
    :param: hm_click: clickData from heatmap.
    :param: admit_type: admission type from selection.
    :param: reset (boolean): reset heatmap graph if True.
    :return: Patient volume annotated heatmap.
    """

#     filtered_df = df[
#         (df["Clinic Name"] == clinic) & (df["Admit Source"].isin(admit_type))
#     ]
    filtered_df = email_df.sort_values("date_es").set_index("date_es")[
        start.astimezone(timezone('US/Eastern')):end.astimezone(timezone('US/Eastern'))
    ]

    filtered_df = filtered_df[filtered_df.weekdays.isin(weekdays)]
    filtered_df = filtered_df[filtered_df.hour.isin(time)]

    x_axis = [datetime.time(i).strftime("%I %p")
              for i in range(24)]  # 24hr time list
    y_axis = day_list

    hour_of_day = ""
    weekday = ""
    shapes = []

    if hm_click is not None:
        hour_of_day = hm_click["points"][0]["x"]
        weekday = hm_click["points"][0]["y"]

        # Add shapes
        x0 = x_axis.index(hour_of_day) / 24
        x1 = x0 + 1 / 24
        y0 = y_axis.index(weekday) / 7
        y1 = y0 + 1 / 7

        shapes = [
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                line=dict(color="#ff6347"),
            )
        ]

    z = np.zeros((7, 24))
    annotations = []

    for ind_y, day in enumerate(y_axis):
        filtered_day = filtered_df[filtered_df["weekdays"] == day]
        for ind_x, x_val in enumerate(x_axis):
            sum_of_record = len(filtered_day[filtered_day["hour"] == x_val])
            z[ind_y][ind_x] = sum_of_record

            annotation_dict = dict(
                showarrow=False,
                text="<b>" + str(sum_of_record) + "<b>",
                xref="x",
                yref="y",
                x=x_val,
                y=day,
                font=dict(family="sans-serif"),
            )
            # Highlight annotation text by self-click
            if x_val == hour_of_day and day == weekday:
                if not reset:
                    annotation_dict.update(size=15, font=dict(color="#ff6347"))

            annotations.append(annotation_dict)

    hovertemplate = "<b> %{y}  %{x} <br><br> %{z} Emails"
    data = [
        dict(
            x=x_axis,
            y=y_axis,
            z=z,
            type="heatmap",
            name="",
            hovertemplate=hovertemplate,
            showscale=False,
            colorscale=[[0, "#caf3ff"], [1, "#2c82ff"]],
        )
    ]

    layout = dict(
        margin=dict(l=70, b=30, t=50, r=50),
        modebar={"orientation": "v"},
        font=dict(family="Open Sans"),
        annotations=annotations,
        shapes=shapes,
        xaxis=dict(
            side="top",
            ticks="",
            ticklen=2,
            tickfont=dict(family="sans-serif"),
            tickcolor="#ffffff",
        ),
        yaxis=dict(
            side="left", ticks="", tickfont=dict(family="sans-serif"), ticksuffix=" "
        ),
        hovermode="closest",
        showlegend=False,
    )
    return {"data": data, "layout": layout}


# In[60]:


def generate_hist(start, end, weekdays, time):

    df = email_df.copy()
    df['date_pure'] = df.date_es.dt.date
    df = df.sort_values("date_pure").set_index("date_pure")

    df['selected'] = False


#     print(df)
#     df.loc[~df.weekdays.isin(weekdays),'selected']=False
#     df.loc[~df.hour.isin(time),'selected']=False

    df.loc[pd.to_datetime(start).date():pd.to_datetime(
        end).date(), 'selected'] = True

    df.loc[~df.weekdays.isin(weekdays), 'selected'] = False
    df.loc[~df.hour.isin(time), 'selected'] = False
    df = df.reset_index()
    fig = px.histogram(df, "date_es", color='selected', marginal="rug", nbins=12,
                       height=400,
                       color_discrete_map={
                           True: "rgb(166,206,227)", False: "rgb(31,120,180)"
                       },
                       )
    fig.update_layout(margin=dict(l=2, r=2, t=2, b=2), height=200)
    fig.layout.xaxis.title.text = None
    fig.layout.yaxis.title.text = 'Count'
#     fig.layout.legend['orientation']='h'
#     fig.update_layout(legend=dict(x=0.25, y=-0.25))
    fig.data[0]['nbinsx'] = 20
    return fig


# In[83]:


# fig=generate_hist('2020-03-15', '2020-06-15', day_list, ['12 PM'])
# fig.layout.xaxis.title.text=None
# fig.layout.margin.l=30
# fig.data


# In[66]:


def haweks(learner, pre_days):

    test_time = (email_df.date_es.sort_values() -
                 email_df.date_es.min()).astype('timedelta64[h]')/24.0
    timestamps = [test_time.to_numpy(dtype='double')]

    best_score = -1e100
    decay_candidates = np.logspace(0, 6, 20)

    if learner == 'Exponential':

        for i, decay in enumerate(decay_candidates):
            hawkes_learner = HawkesExpKern(decay, verbose=False, max_iter=10000,
                                           tol=1e-10)
    #         hawkes_learner = HawkesSumExpKern(decays=[6])
            hawkes_learner.fit(timestamps)

            hawkes_score = hawkes_learner.score()
            if hawkes_score > best_score:
                print('obtained {}\n with {}\n'.format(hawkes_score, decay))
                best_hawkes = hawkes_learner
                best_score = hawkes_score

    elif learner == 'ExponentialSum':
        decay_candidates = np.logspace(0, 3, 10)
        for i, decays in enumerate(itertools.combinations(decay_candidates, 3)):
            # Each time we test a different set of 3 decays.
            decays = np.array(decays)
            hawkes_learner = HawkesSumExpKern(decays, verbose=False, max_iter=10000,
                                              tol=1e-10)
#             hawkes_learner._prox_obj.positive = False
            hawkes_learner.fit(timestamps)

            hawkes_score = hawkes_learner.score()
            if hawkes_score > best_score:
                print('obtained {}\n with {}\n'.format(hawkes_score, decays))
                best_hawkes = hawkes_learner
                best_score = hawkes_score

    simu = best_hawkes._corresponding_simu()
    simu.seed = 2020
    simu.track_intensity(0.01)
    simu.set_timestamps([test_time.to_numpy(dtype='double')])
    simu.end_time = 100+pre_days
    simu.simulate()


# process = plot_point_process(simu, plot_intensity=True)

    plotly_fig = tls.mpl_to_plotly(
        plot_point_process(simu, plot_intensity=True))

    return seperate(plotly_fig)


# In[63]:


def seperate(f):
    original_f = go.Figure(f)
    cop_f = go.Figure(f)

    new_color = 'rgba(63, 191, 63, 0.4)'

    cop_f.data[0]['x'] = tuple(filter(lambda x: x > 100, cop_f.data[0]['x']))
    cop_f.data[0]['y'] = cop_f.data[0]['y'][len(
        cop_f.data[0]['y'])-len(cop_f.data[0]['x']):]

    cop_f.data[1]['x'] = tuple(filter(lambda x: x > 100, cop_f.data[1]['x']))
    cop_f.data[1]['y'] = cop_f.data[1]['y'][len(
        cop_f.data[1]['y'])-len(cop_f.data[1]['x']):]

    cop_f.data[1]['marker']['color'] = new_color
    cop_f.data[1]['marker']['line']['color'] = new_color

    cop_f.data[0]['line']['color'] = new_color

    original_f.data[0]['x'] = tuple(
        filter(lambda x: x <= 100, original_f.data[0]['x']))

    original_f.data[0]['y'] = original_f.data[0]['y'][:len(
        original_f.data[0]['x'])]

    original_f.data[1]['x'] = tuple(
        filter(lambda x: x <= 100, original_f.data[1]['x']))
    original_f.data[1]['y'] = original_f.data[1]['y'][:len(
        original_f.data[1]['x'])]

    original_f.add_traces([i for i in cop_f['data']])

    original_f.update_layout({'height': 400})
#     original_f.layout.margin['l']=25
    original_f.layout.margin = {
        'b': 50, 'l': 25, 'r': 30, 't': 50
    }

    original_f.layout['xaxis']['title'] = {'font': {
        'color': '#000000', 'size': 13.0}, 'text': 'No. of Days since the 1st Rej Letter'}

    original_f.update_layout(showlegend=True)
    original_f.update_layout(legend_orientation="h")
    original_f.update_layout(legend=dict(x=0.25, y=-0.25))

    original_f.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=100,
            y0=0,
            x1=100,
            y1=3,
            line=dict(
                color="RoyalBlue",
                width=2,
                 dash="dashdot"
            )
        ))

    original_f['data'][0]['name'] = 'Estimated Intensity of Original Events'
    original_f['data'][1]['name'] = 'Original Events'

    original_f['data'][2]['name'] = 'Estimated Intensity of Predicted Events'
    original_f['data'][3]['name'] = 'Predicted Events'

    original_f.layout['title'] = {'font': {'color': 'rgb(87, 145, 203)', 'size': 17},
                                  'text': 'Hawekes Modelling Results', 'xanchor': 'center',
                                  'yanchor': 'top', 'x': 0.5}
    original_f.layout.margin.l = 30
#     original_f.layout.width=1000
    original_f.layout.autosize = True

    return original_f


# In[82]:


def cal_slider(start_date, end_date):
    start_value = (dt.strptime(start_date, "%Y-%m-%d") -
                   (email_df['date_es'].min().tz_convert(None))).days+1
    time_delta = (dt.strptime(end_date, "%Y-%m-%d")) - \
        (dt.strptime(start_date, "%Y-%m-%d"))
    end_value = time_delta.days
    return [start_value, start_value+end_value]


def cal_range(value):
    start_value, end_value = value
    start_date = email_df['date_es'].min().date() + \
        datetime.timedelta(start_value)
    end_date = start_date+datetime.timedelta(end_value)
    return start_date, end_date


app = dash.Dash(__name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css",
                                                "https://dash-gallery.plotly.host/dash-oil-and-gas/assets/styles.css?m=1590087908.0"])
# app = JupyterDash(__name__,external_stylesheets=["https://dash-gallery.plotly.host/dash-oil-and-gas/assets/styles.css?m=1590087908.0"])


app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner"
        ),  # Left column
        html.Div(
            id="left-column ",
            style={},
            className="four columns pretty_container ",
            children=[intro(), control_card()],
        ),
        html.Div(
            id="right-column",
            className="eight columns",
            style={},
            children=[html.Div(
                [
                    html.Div(
                        [html.H6(),
                         html.P("No. of Days Selected"),
                         html.Strong(id="days_selected")],
                        id="days",
                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(), html.P(
                            "No. of Letters Received in the Period"),
                         html.Strong(id="total")],

                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(), html.P(
                            "Peak Day and Hour"),
                         html.Strong(id="peak_date")],

                        className="mini_container",
                    ),
                    html.Div(
                        [html.H6(id="pn"), html.P(
                            "Rej Letters Peak Volume"),
                         html.Strong(id="peak_num")],

                        className="mini_container",
                    ),
                ],
                id="info-container",
                className="row container-display",
            ),
                # Patient Volume Heatmap
                html.Div(id='prop_id'),
                html.Div(id='prop_type'),
                html.Div(id='prop_value'),

                html.Div(
                id="patient_volume_card",
                className='mini_container',
                children=[
                    dcc.Loading(dcc.Graph(id='hist')),

                    html.Hr(), html.B("Email Heatmap"),
                    html.Div(
                        'The traffic of rejection emails! Click on the cells to see the actual entry.'),
                    dcc.Graph(id="patient_volume_hm"),
                    dcc.RangeSlider(
                        id='datetime_RangeSlider',
                        updatemode='mouseup',  # don't let it update till mouse released
                        min=0,
                        #                 disabled=True,
                        max=(email_df['date_es'].max().date() - email_df['date_es'].min().date()).days),
                    html.Div(id='table_div', style={'display': 'none'},
                             children=[dash_table.DataTable(
                                 id='table',
                                 style_cell={'textAlign': 'left', 'padding': '5px',
                                             'overflow': 'hidden',
                                             'textOverflow': 'ellipsis'},
                                 style_data={'whiteSpace': 'normal'},
                                 css=[{
                                     'selector': '.dash-cell div.dash-cell-value',
                                     'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                                 }],
                                 columns=[
                                     {'name': i, 'id': i, 'deletable': True} for i in ['date_es', 'subject']
                                 ],
                                 page_current=0,
                                 #                                  page_size=1,
                                 #                                  page_action='custom',

                                 #                                  sort_action='custom',
                                 #                                  sort_mode='single',
                                 sort_by=[])],

                             )
                ],
            )]),
        html.Div(id='para', className="columns pretty_container", children=[
            html.H5('Parallel Coordinates of the Flow of Rej. Emails.'),
            html.Div("Let's highlight the most prominent streamline of the email flow. \
                     It could come in handy when we observed some trends in the data. \
            This plot will update automatically with the Date/Day/Hour you selected in the Control widgets.",
                     style={'margin': "10px"}),
            dcc.Graph(id='paco')]
        ),
        html.Div(id='temperal pro', className='columns pretty_container', children=[

            html.H5("Temporal Process Analytics",
                    style={'margin-left': '10px'}),
            html.P(
                children="A Temporal Process is a kind of random process whose realization consists of discrete events \
                localized in time. Compared with \
                traditional Time-Seris, each data entry was allocated in different time interval. The scattering nature of receiving\
                an email fits better with a Temporal Process Analysis. \n \
                A very popular kind of termporal process is the Haweks process, which could be consider\
                as an 'auto-regression' type of process. Here I used the Haweks Process to simulate the events.\
                You can select the Kernal and the days to forecast below.",
                style={'margin': '10px'}
            ),
            html.Div(
                id="select model",
                style={"text-align": "center"},
                className="six columns", children=[html.Strong('Select the Kernal'),
                                                   dcc.RadioItems(
                                                       id='modelpicker',
                                                       options=[
                                                           {'label': 'Exponential Kernel',
                                                               'value': 'Exponential'},
                                                           {'label': 'ExponentialSum Kernel',
                                                            'value': 'ExponentialSum'}
                                                       ],
                    value='Exponential'
                )

                ]),
            html.Div(
                id="select days",
                style={"text-align": "center"},
                className="six columns", children=[html.Strong('Select the # of Days in the future to Predict'),
                                                   dcc.Slider(
                    id='daysslider',
                    min=1,
                    max=100,
                    step=1,
                    value=10,
                    updatemode='drag'
                )
                ]),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Div(className='nine columns', children=dcc.Loading(
                dcc.Graph(id='processline'))),
            html.Div(className='three columns container', children=[
                html.Div(className='container', children=[html.Div(
                    [html.H6(), html.P(
                        "  No. Days After the Last Rej Letter that I Received"),
                        html.Strong(id="days_after")],

                    className="mini_container",
                ), html.Br(), html.Br(),
                    html.Div(
                        [html.H6(), html.P(
                            'Exact Time of the Email (Received/To be Received)'),
                         html.Strong(id="exact_time")],

                        className="mini_container",
                )


                ]
                )




            ]),



        ]),

        html.Div(className='columns mini_container', children=[
            html.H5("Email Content Analysis", style={'margin': '10px'}),
            html.Div("After cleaning the text of the emails, we can find out what words or phrases are the important or interesting .\
            I provided two commonly used metrics for you to rank the words/phrases. On the right panel, I have tried to present you with\
            the interesting bigrams, a.k.a. word collocations. Feel free to change the metric to see \
            which words are connected.", style={'margin': '10px'}),
            html.P('P.S. It might take a long time for the left graph to show up.', style={
                   'margin': '10px'}),
            html.Div(id='phrase', className='six columns',
                     style={'text-align': 'center'},
                     children=[
                         html.Strong(
                             'Select the Metric and the # of Words to Show'),
                         html.Div(children=[dcc.Dropdown(
                             id='tf_selector',
                             options=[
                                 {'label': 'Word Count(Term Frequency)',
                                  'value': 'tf'},
                                 {'label': 'Term Frequency-Inverse Document Frequency(TF-IDF)', 'value': 'tfidf'}],
                             value='tf'
                         ),

                             dcc.Dropdown(
                             id='rank_selector',
                             options=[
                                 {'label': '10', 'value': 10},
                                 {'label': '15', 'value': 15},
                                 {'label': '20', 'value': 20}],
                             value=10
                         )]),
                         html.Br(),
                         dcc.Loading(dcc.Graph(id='barchart')),
                         dcc.Loading(dcc.Graph(id='treechart'))



                     ]),

            html.Div(style={'margin-left': '7%', 'text-align': 'center'}, className='five columns', children=[
                html.Strong('Select the Interestingness Metric'),
                dcc.RadioItems(
                    id='colmetric',
                    options=[
                        {'label': 'Point-Wise Mutual Infomation', 'value': 'pmi'},
                        {'label': 'Chi-Square', 'value': 'chisquare'},
                        {'label': 'Likelihood Ratio', 'value': 'likelihood_ratio'}
                    ],
                    value='pmi'),
                html.Div(id='cos'),
                html.Br(),
                html.Br(),
                dash_table.DataTable(
                    id='co_table',
                    style_cell={'textAlign': 'left', 'padding': '5px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis'},
                    style_data={'whiteSpace': 'normal'},
                    css=[{
                        'selector': '.dash-cell div.dash-cell-value',
                        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                    }],
                    columns=[
                        {'name': i, 'id': i, 'deletable': True} for i in ['collacation', 'metric']
                    ],
                    page_current=0,
                    page_size=1,
                    page_action='custom',

                    sort_action='custom',
                    sort_mode='single',
                    sort_by=[])


            ])

        ]),
        html.Div(className='mini_container columns', children=[
            #             html.H5('Topic Modelling'),

            html.Div(id='clickdata'),
            html.Div(className='columns', children=[
                html.Div(className='six columns',
                         children=[html.Div(className='six columns container',
                                            style={
                                                'width': "70%",
                                                'margin': "30px"},
                                            children=[html.H5('Topic Modelling'), html.Br(), html.Blockquote('We can further \
                                            explore what these emails are mainly talking about by applying topic modeling techniques to \
            the texts. To achieve the best results, the texts were cleaned by removing extra elements(like HTML tags), punctuations, and numbers.\
            The stopwords are removed as well. I used the NLTK stopword collection and extended it with other self-defined words, like my name😂. \
            Then, the n-grams(I only used the uni-, bi- and, tri-grams here.) are generated. As an unsupervised learning method, \
            the number of topics should be specified, here the number is automatically selected by maximizing the coherence values; however, \
            you can change the number of topics by clicking the dots in the line chart.')])]),

                html.Div(className='six columns',
                         style={'text-align': 'center'},
                         children=[
                             html.Div(style={'display': 'inline-flex'}, children=[html.Div('Number of Topics Selected', style={
                                      'margin-right': '20px'}), dcc.Input(id="current_k", value=4, disabled=True)]),
                             #                              style={'display':'inline-flex'},
                             dcc.Graph(id='coherence',
                                       figure=px.line(choose_k, x='# of Topics', y='coherence').update_traces(
                                           mode='lines+markers')
                                       )]
                         ),
                #                 html.Hr(),
                html.Br(),
                 html.H6("Let's see the temporal distribution and the linguistic features of these topics.",
                         style={'text-align': 'center', 'width': "100%", 'margin-top': '20px'}, className='columns'),
                 html.Div(className='six columns', children=[
                     dcc.Loading(dcc.Graph(id='sunburst'))]),
                 #                 html.Div(style={"border-left":"1px solid #000","height":"500px"}),
                 html.Div(className='six columns', children=[
                     dcc.Loading(dcc.Graph(id='polar'))]),
                 #                 html.Hr(),
                 html.Br(),
                 html.Br(),
                 html.H6('Visualizing the Topic Modeling Results with t-SNE',
                         style={'text-align': 'center', 'width': "100%", 'margin-top': '20px'}, className='columns'),
                 html.Hr(),
                 html.Div(className='columns',
                          children=[dcc.Loading(dcc.Graph(id='tsne'))])
                 ])]),
        html.Div(id='contact_info', style={'text-align': 'center'}, className='pretty_container twelve columns', children=[
            'Thanks for playing with it! You can contact me via my ',
            html.A(
                'LinkedIn', href='https://www.linkedin.com/in/yukun-yang-1044ab157/', target="_blank"),
            ', or ',
            html.A('Personal Website',
                   href='http://www.yukunyang.info', target="_blank"),
            '. You can also Email me at ',
            html.A('contact@yukunyang.info',
                   href='mailto:contact@yukunyang.info', target="_blank")
        ])




    ])


@app.callback(
    Output('current_k', 'value'),
    [Input('coherence', 'clickData')])
def click_to_change_k(clickData):

    if clickData == None:
        return 4
    else:
        return clickData['points'][0]['x']


@app.callback(
    [Output('sunburst', 'figure'),
     Output('polar', 'figure'),
     Output('tsne', 'figure')],
    [Input('current_k', 'value')])
def change_of_k(k):
    lda_model = model_list[int((k-2)/2)]

    df_topic_sents_keywords = format_topics_sentences(
        ldamodel=lda_model, corpus=corpus, texts=data_lemmatized)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = [
        'Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    topic_email = pd.merge(how='left', left=email_df.dropna().reset_index(
    ), right=df_dominant_topic, left_index=True, right_index=True)
    topic_email['Month'] = topic_email['date_es'].dt.month.map(
        {3: "March", 4: "April", 5: 'May', 6: 'June'})

    topic_email['lexicon_count'] = topic_email.cleaned.apply(
        textstat.lexicon_count, removepunct=True)
    topic_email['reading_ease'] = topic_email.extracted.apply(
        textstat.flesch_reading_ease)
    topic_email['unique_term_count'] = topic_email.cleaned.apply(
        lexicalrichness.LexicalRichness).apply(lambda x: x.terms)
    topic_email['lexical_diversity'] = topic_email.cleaned.apply(
        lexicalrichness.LexicalRichness).apply(lambda x: x.mtld(threshold=0.72))

    month_topic_dis = topic_email.groupby(['Month', 'Dominant_Topic'])[
        'date'].count().reset_index().rename(columns={'date': 'Count'})

    categories = ['lexicon_count', 'reading_ease',
                  'unique_term_count', 'lexical_diversity']
    melted_df = pd.melt(topic_email.groupby('Dominant_Topic')[categories].mean().reset_index(), id_vars=['Dominant_Topic'], value_vars=categories
                        )
    polar = px.line_polar(melted_df, r="value", theta="variable",
                          color="Dominant_Topic", line_close=True, title='Linguistic Features of Topics')
    sun = px.sunburst(month_topic_dis, path=['Month', 'Dominant_Topic'], values='Count',
                      color='Count', title='Monthly Topic Distribution', color_continuous_scale=px.colors.sequential.Blues)
    polar.update(layout=dict(title=dict(x=0.5)))
    sun.update(layout=dict(title=dict(x=0.5)))
    from sklearn.manifold import TSNE
    topic_weights = []

    topic_weights = pd.DataFrame()

    for i, row_list in enumerate(lda_model[corpus]):
        #     print(i,row_list)
        for j in row_list:
            topic_weights.loc[i, j[0]] = j[-1]
    #     topic_weights.append([row_list[0][-1]])

    arr = topic_weights.fillna(0).values
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1,
                      random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    tsne_df = pd.DataFrame(tsne_lda)
    tsne_df = tsne_df.rename(columns={0: 'x', 1: 'y'})
    tsne_df['Dominant_Topic'] = topic_num
    tsne_df = pd.merge(left=tsne_df, right=df_dominant_topic,
                       left_on='Dominant_Topic', right_on='Dominant_Topic')
    tsne_df['Dominant_Topic'] = tsne_df['Dominant_Topic'].apply(
        lambda x: 'Topic'+str(x))
    tsne = px.scatter(tsne_df, x='x', y='y', color='Dominant_Topic',
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      hover_data=['x', 'y', 'Keywords'])
#     tsne.update_traces(hovertemplate= "x:%{x}: <br>y: %{y} </br> Topic Keywords:%{hover_data}")
    return sun, polar, tsne


@app.callback([dash.dependencies.Output('cos', 'children'),
               dash.dependencies.Output('co_table', 'data')],
              [dash.dependencies.Input('colmetric', 'value')]
              )
def update_cos(metric):

    co_list = collo(metric)
    table = pd.DataFrame(co_list, columns=['collacation', 'metric'])

    top_10 = table.head(10)
    top_10['collacation'] = top_10.apply(
        lambda x: ' '.join(word for word in x['collacation']), axis=1)

    top_10 = top_10.to_dict('record')
    return[make_circos(co_list)], top_10


@app.callback([dash.dependencies.Output('barchart', 'figure'),
               dash.dependencies.Output('treechart', 'figure')],
              [dash.dependencies.Input('tf_selector', 'value'),
               dash.dependencies.Input('rank_selector', 'value')]
              )
def update_import_words(metric, rank):

    return make_important_graphs(important_words(metric, rank))


@app.callback([dash.dependencies.Output('daysslider', 'marks')],
              [dash.dependencies.Input('daysslider', 'value')])
def add_marks(value):
    return [{value: {'label': str(value)+' days'}}]


@app.callback([dash.dependencies.Output('days_after', 'children'),
               dash.dependencies.Output('exact_time', 'children')],
              [dash.dependencies.Input('processline', 'hoverData')])
def conver_point(hoverData):

    if hoverData == None:
        return ('Please Hover over the Dots in the Graph', 'Please Hover over the Dots in the Graph')

    if (hoverData["points"][0]["curveNumber"] == 1) or (hoverData["points"][0]["curveNumber"] == 3):
        days_passed = hoverData["points"][0]['x']
        hours_passed = days_passed*24
        exact_date = email_df.date_es.min()+datetime.timedelta(hours=hours_passed)

        return round(days_passed, 2), exact_date.strftime("%m/%d/%Y, %H:%M:%S")
    else:
        return no_update, no_update


@app.callback([dash.dependencies.Output('processline', 'figure')],
              [dash.dependencies.Input('modelpicker', 'value'),
               dash.dependencies.Input('daysslider', 'value')])
def update_process(model, days):

    return [haweks(model, days)]


@app.callback([dash.dependencies.Output('table_div', 'style'),
               dash.dependencies.Output('table', 'data')],
              [dash.dependencies.Input('patient_volume_hm', 'clickData')],
              [dash.dependencies.State('my-date-picker-range', 'start_date'),
               dash.dependencies.State('my-date-picker-range', 'end_date'),
               dash.dependencies.State('weekdays', 'value'),
               dash.dependencies.State('time', 'value')])
def test(heatmap_click, start_date, end_date, weekdays, time):

    if heatmap_click == None:
        return no_update
    else:
        ctx = dash.callback_context

        prop_id = ""
        prop_type = ""
        triggered_value = None
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            prop_type = ctx.triggered[0]["prop_id"].split(".")[1]
            triggered_value = ctx.triggered[0]["value"]

            filtered = filter_df(dt.strptime(
                start_date, "%Y-%m-%d"), dt.strptime(end_date, "%Y-%m-%d"), weekdays, time)
            hour_of_day = heatmap_click["points"][0]["x"]
            weekday = heatmap_click["points"][0]["y"]

            click_df = filtered[filtered.hour == hour_of_day]
            click_df = click_df[click_df.weekdays == weekday]

        return ({'display': 'block'}, click_df.to_dict('record'))


# @dcb.callback([Output("input", "value"), Output("slider", "value")], [Input("sync", "data")],
#               [State("input", "value"), State("slider", "value")])
# def update_components(current_value, input_prev, slider_prev):
#     # Update only inputs that are out of sync (this step "breaks" the circular dependency).
#     input_value = current_value if current_value != input_prev else dash.no_update
#     slider_value = current_value if current_value != slider_prev else dash.no_update
#     return [input_value, slider_value]


@app.callback(
    [dash.dependencies.Output('patient_volume_hm', 'figure'),
     dash.dependencies.Output('datetime_RangeSlider', 'value'),
     dash.dependencies.Output('datetime_RangeSlider', 'marks'),
     dash.dependencies.Output('hist', 'figure'),
     dash.dependencies.Output('days_selected', 'children'),
     dash.dependencies.Output('total', 'children'),
     dash.dependencies.Output('peak_date', 'children'),
     dash.dependencies.Output('peak_num', 'children'),
     dash.dependencies.Output('paco', 'figure')
     ],
    [dash.dependencies.Input('my-date-picker-range', 'start_date'),
     dash.dependencies.Input('my-date-picker-range', 'end_date'),
     dash.dependencies.Input('weekdays', 'value'),
     dash.dependencies.Input('time', 'value'),
     dash.dependencies.Input('patient_volume_hm', 'clickData')])
def update_output_from_picker(start_date, end_date, weekdays, time, click):
    heatmapdata = generate_patient_volume_heatmap(dt.strptime(
        start_date, "%Y-%m-%d"), dt.strptime(end_date, "%Y-%m-%d"), click, None, weekdays, time)
    hist_plot = generate_hist(start_date, end_date, weekdays, time)

    ind = np.unravel_index(np.argmax(
        heatmapdata['data'][0]['z'], axis=None), heatmapdata['data'][0]['z'].shape)
    day = heatmapdata['data'][0]['y'][ind[0]]
    hour = heatmapdata['data'][0]['x'][ind[-1]]

    filtered = filter_df(dt.strptime(
        start_date, "%Y-%m-%d"), dt.strptime(end_date, "%Y-%m-%d"), weekdays, time)

    return (heatmapdata,
            cal_slider(start_date, end_date),
            {0: email_df['date_es'].min().date(), cal_slider(start_date, end_date)[0]: start_date, cal_slider(
                start_date, end_date)[-1]: end_date, 100: email_df['date_es'].max().date()},
            hist_plot,
            (dt.strptime(end_date, "%Y-%m-%d") -
             dt.strptime(start_date, "%Y-%m-%d")).days,
            heatmapdata['data'][0]['z'].sum(),
            day+' '+hour,
            heatmapdata['data'][0]['z'].max(),
            generate_paco(filtered))


if __name__ == '__main__':

    app.run_server()


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:

