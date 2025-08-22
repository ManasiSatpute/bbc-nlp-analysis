# bbc-nlp-analysis
NLP preprocessing, POS tagging, and Named Entity Recognition on BBC news headlines using NLTK &amp; spaCy
#  BBC News NLP Analysis

This project applies **Natural Language Processing (NLP)** techniques on the **BBC News dataset** to explore text preprocessing, Part-of-Speech (POS) tagging, and Named Entity Recognition (NER).  
It uses **NLTK** and **spaCy** for preprocessing and linguistic analysis.  



##  Features
- Lowercasing, stopword removal, punctuation removal  
- Tokenization (raw + cleaned tokens)  
- Lemmatization using `WordNetLemmatizer`  
- Part-of-Speech tagging with spaCy  
- Frequency analysis of nouns, verbs, adjectives  
- Named Entity Recognition (NER) with spaCy  
- Aggregated counts of tokens and entities  



## 🛠 Tech Stack
- Python 3
- Pandas
- NLTK
- spaCy
- Matplotlib (optional for visualization)



## 📊Example Outputs

###  POS Tag Frequency (Top 10)
| Token   | POS  | Count |
|---------|------|-------|
| news    | NOUN | 120   |
| say     | VERB | 90    |
| new     | ADJ  | 75    |

###  Named Entities
| Entity          | Label   | Count |
|-----------------|---------|-------|
| London          | GPE     | 25    |
| United Kingdom  | GPE     | 20    |
| Apple           | ORG     | 15    |



##  Dataset
The dataset used is `bbc_news.csv`.  
If not provided, you can download a similar version here: [BBC Text Classification Dataset](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category).  



