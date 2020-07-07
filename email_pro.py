import mailbox
import email.utils

import pandas as pd
import numpy as np
import datetime
import plotly.express as px

import re, string

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import nltk
from nltk.corpus import stopwords
import gensim
import spacy
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

    
import mailbox
import bs4

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts, stop):

    new_docs=[]
    for doc in texts:
        new_docs.append([word for word in doc if word not in stop])
    return new_docs    
    
def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def extract_text(text_list):
    
    tags=[component[0] for component in text_list]
    
    real_content=None
    if 'text/html' in tags:
        ind=[component[0] for component in text_list].index('text/html')
        real_content=text_list[ind][-1]
        if real_content=='None':
            real_content=None

    elif 'text/plain' in tags:
        ind=[component[0] for component in text_list].index('text/plain')
        real_content=text_list[ind][-1]
    elif 'NA' in tags:
        ind=[component[0] for component in text_list].index('NA')
        real_content=text_list[ind][-1]
    
    if (real_content is not None):
            if len(real_content)>10000:
                real_content=None
    
    return real_content
  
        
def clean_text(text):
    
    if text is not None:
    
        text=re.sub('=\n', '', text) 

        text=re.sub('\S*@\S*\s?', '',  text)

        text=' '.join(word.strip(string.punctuation) for word in text.split())

        text=re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '', text, flags=re.MULTILINE)

    #     text=re.sub(r'\..*\..* ?', '', text, flags=re.MULTILINE)

        text=re.sub(r"\d+", "", text)

        text=re.sub(r"={1}.{2}", "", text)
        
        text=text.replace('size','').replace('text size', '').replace('adjust','').replace('td','')
        
        
        
        return text
    else:
        return None

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
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,random_state=2020)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def get_html_text(html):
        try:
            return bs4.BeautifulSoup(html, "html5lib").body.get_text(' ', strip=True)
        except AttributeError: # message contents empty
            return None

class GmailMboxMessage():
        def __init__(self, email_data):
            if not isinstance(email_data, mailbox.mboxMessage):
                raise TypeError('Variable must be type mailbox.mboxMessage')
            self.email_data = email_data
            self.labels= self.date= self.efrom= self.eto= self.subject= self.text=None
            

        def parse_email(self):
            email_labels = self.email_data['X-Gmail-Labels']
            email_date = self.email_data['Date']
            email_from = self.email_data['From']
            email_to = self.email_data['To']
            email_subject = self.email_data['Subject']
            email_text = self.read_email_payload() 
            
            self.labels=email_labels
            self.date=email_date
            self.efrom=email_from
            self.eto=email_to
            self.subject=email_subject
            self.text=email_text
            

        def read_email_payload(self):
            email_payload = self.email_data.get_payload()
            if self.email_data.is_multipart():
                email_messages = list(self._get_email_messages(email_payload))
            else:
                email_messages = [email_payload]
            return [self._read_email_text(msg) for msg in email_messages]

        def _get_email_messages(self, email_payload):
            for msg in email_payload:
                if isinstance(msg, (list,tuple)):
                    for submsg in self._get_email_messages(msg):
                        yield submsg
                elif msg.is_multipart():
                    for submsg in self._get_email_messages(msg.get_payload()):
                        yield submsg
                else:
                    yield msg

        def _read_email_text(self, msg):
            content_type = 'NA' if isinstance(msg, str) else msg.get_content_type()
            encoding = 'NA' if isinstance(msg, str) else msg.get('Content-Transfer-Encoding', 'NA')
            if 'text/plain' in content_type and 'base64' not in encoding:
                msg_text = msg.get_payload()
            elif 'text/html' in content_type and 'base64' not in encoding:
                msg_text = get_html_text(msg.get_payload())
            elif content_type == 'NA':
                msg_text = get_html_text(msg)
            else:
                msg_text = None
            return (content_type, encoding, msg_text)

def main():
        

    mbox = mailbox.mbox('Rej.mbox')
    mbox2 = mailbox.mbox('rej2.mbox')


    emails=[]
    num_entries = len(mbox)
    for idx, email_obj in enumerate(mbox):
        email_data = GmailMboxMessage(email_obj)
        email_data.parse_email()
        emails.append(email_data)


    num_entries = len(mbox2)
    for idx, email_obj in enumerate(mbox2):
        email_data = GmailMboxMessage(email_obj)
        email_data.parse_email()
        emails.append(email_data)

    email_df= pd.DataFrame()
    for e in emails:
        email_df=email_df.append([{'date':e.date,'from':e.efrom,'to':e.eto,'subject':e.subject,'text':e.text}])


    from pytz import timezone


    from dateutil.parser import parse
    from datetime import datetime as dt



    email_df['date_n']=pd.to_datetime(email_df.date)

    email_df['date_es']=email_df['date_n'].apply(lambda x: x.astimezone(timezone('US/Eastern')))


    email_df['weekdays']=email_df.date_es.apply(lambda x: dt.strftime(x, "%A"))

    email_df['hour']=email_df.date_es.apply(
        lambda x: dt.strftime(x, "%I %p")
    ) 


    nltk.download('stopwords')
    stop = list(stopwords.words('english'))
    stop.extend(['yukun','yukun yang','yang','data','scientist'])

  

    email_df['extracted']=email_df.text.apply(extract_text)
    email_df['cleaned']=email_df.extracted.apply(clean_text)

    data=email_df[email_df.cleaned.notna()].cleaned.values.tolist()

    data_words = list(sent_to_words(data))

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words, stop)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)

    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, nlp,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # print(id2word)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    print('generating models')
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=2)



    x=range(2,40,2)

    choose_k=pd.DataFrame({'# of Topics':x,'coherence':coherence_values})

    print("topic testing")


    choose_k.to_csv("choose_k.csv",index=False)
    email_df.to_csv("email_to_df.csv", index=False)

    import pickle
    with open('model_list.pickle', 'wb') as b:
        pickle.dump(model_list,b)

    with open('id2word.pickle', 'wb') as b:
        pickle.dump(id2word,b)

    with open('texts.pickle', 'wb') as b:
        pickle.dump(texts,b)

    with open('corpus.pickle', 'wb') as b:
        pickle.dump(corpus,b)
    

if __name__ == '__main__':
    main()
    # choose_k.to_csv("choose_k.csv",index=False)
    # email_df.to_csv("email_to_df.csv", index=False)

