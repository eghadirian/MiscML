from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import nltk
import re
import numpy as np

# countvectorizer
reviews_train = load_files('C:/Users/Emad/.PyCharmCE2019.3/config/scratches/Data Set/aclImdb_v1/aclImdb/train')
# download from http://ai.stanford.edu/~amaas/data/sentiment/
text_train, y_train = reviews_train.data, reviews_train.target
text_train = [doc.replace(b'<br />', b' ') for doc in text_train]
vect = CountVectorizer(min_df=5, stop_words='english').fit(text_train)
X_train = vect.transform(text_train)
# tf-idf: term-frequency inverse-document frequency
pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
param_grid = {'logisticregression__C': [10**(i-3) for i in range(6)],
              'tfidfvectorizer__ngram_range':[(1,1),(1,2),(1,3)]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
# lemmatization
en_nlp = spacy.load('en')
# stemmer
stemmer = nltk.stem.PorterStemmer()
regexp = re.compile('(?u)\\b\\w\\w+\\b') # **wild cards**
old_tokenizer = en_nlp.tokenizer
# **understand this**
en_nlp.tokenizer = lambda string:\
    old_tokenizer.tokens_from_list(regexp.findall(string))
def custom_tokenizer(document):
    doc_spacy = en_nlp(document, entity=False, parse=Flase)
    return [token.lemma_ for token in doc_spacy]
lemma_vect = CountVectorizer(tokenizer=custom_tokenizer(), min_df=5)
X_train_lemma = lemma_vect.fit_transform(text_train)
X_train = vect.transform(text_train)
# LDA: Latent Dirichlet Allocation
lda = LatentDirichletAllocation(n_topics=100, learning_method='batch', max_iter=25, random_state=0)
document_topics = lda.fit_transform(X_train)
topics = np.array([7])
sorting = np.argsort(lda.components_, axis=1)[:,::-1]
feature_names = np.array(vect.get_feature_names())
print(topics, feature_names)
