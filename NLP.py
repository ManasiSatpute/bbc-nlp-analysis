import nltk 
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt


# LOAD DATA

bbc_data = pd.read_csv('bbc_news.csv')
bbc_data.head()
bbc_data.info()

# By using dataframe from pd we can create column 
titles = pd.DataFrame(bbc_data['title'])
titles.head()


# CLEANING TITLE DATA


# 1. Lowercase
titles['lowercase'] = titles['title'].str.lower()
titles.head()

# 2. Remove stopwords 
en_stopwords = stopwords.words('english')
titles['stopwords'] = titles['lowercase'].apply(
    lambda x:' '.join([word for word in x.split() if word not in en_stopwords]))

# 3. Remove punctuation
titles['Review_no_punctuation'] = titles.apply(
    lambda x: re.sub(r"([^\w\s])", "", x['stopwords']),
    axis=1)

# 4. Tokenization
titles['tokens_raw'] = titles.apply(lambda x: word_tokenize(x['title']), axis=1)
titles['token_row_clean'] = titles.apply(lambda x: word_tokenize(x['Review_no_punctuation']), axis=1)

# 5. Lemmatization
lemmatizer = WordNetLemmatizer()
titles['lemminization'] = titles.apply(
    lambda x: [lemmatizer.lemmatize(word) for word in x['token_row_clean']], axis=1
)

titles.head()

# CREATE TOKEN LISTS

token_raw_list = sum(titles['tokens_raw'], [])
token_clean_list = sum(titles['lemminization'], [])


# SPACY POS TAGGING

nlp = spacy.load('en_core_web_sm')
spacy_doc = nlp(' '.join(token_raw_list))

# Create empty DataFrame
pos_df = pd.DataFrame(columns=['token', 'pos_tag'])

# Extract tokens + POS tags
for token in spacy_doc:
    pos_df = pd.concat(
        [pos_df, pd.DataFrame.from_records([{'token': token.text, 'pos_tag': token.pos_}])],
        ignore_index=True
    )

# Group + count most frequent tokens
pos_df_count = (
    pos_df.groupby(['token', 'pos_tag'])
    .size()
    .reset_index(name='counts')
    .sort_values(by='counts', ascending=False)
)

pos_df_count.head(10)

nouns = pos_df_count[pos_df_count.pos_tag == 'NOUN']
nouns.head(10)

verbs = pos_df_count[pos_df_count.pos_tag == 'VERB']
verbs.head(10)

adj = pos_df_count [pos_df_count.pos_tag=='ADJ']
adj.head(10)

ner_df = pd.DataFrame(columns=['token' ,'ner_tag'])
for token in spacy_doc.ents:
  if pd.isna(token.label_) is False:
    ner_df = pd.concat(
        [ner_df , pd.DataFrame.from_records([{'token':token.text , 'ner_tag':token.label_}])],
        ignore_index=True
    )
ner_df.head(10)

ner_df_count = ner_df.groupby(['token' , 'ner_tag']).size().reset_index(name='counts').sort_values(by='counts' , ascending=False)
ner_df_count.head(10)