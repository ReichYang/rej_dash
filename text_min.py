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



def main():

    nltk.download('stopwords')
    stop = list(stopwords.words('english'))
    stop.extend(['yukun','yukun yang','yang','data','scientist'])

    email_df = pd.read_csv("email_to_df.csv")

    email_df['extracted']=email_df.text.apply(extract_text)
    email_df['cleaned']=email_df.extracted.apply(clean_text)

    data=email_df[email_df.cleaned.notna()].cleaned.values.tolist()

    data_words = list(sent_to_words(data))

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    print(id2word)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=2)



    x=range(2,40,2)

    choose_k=pd.DataFrame({'# of Topics':x,'coherence':coherence_values})
    return choose_k

    

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):

    new_docs=[]
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

# In[16]:

if __name__ == '__main__':
    print(main())
