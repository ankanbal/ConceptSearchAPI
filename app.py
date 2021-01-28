from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

import re
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from flask import Flask, jsonify, abort, make_response, request
from flask_cors import CORS, cross_origin

import gensim
from gensim.utils import unicode
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import LsiModel, Word2Vec, FastText
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import nltk
nltk.download('punkt')


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# loading the models for synonym search
# w = models.KeyedVectors.load_word2vec_format(r'newEnron')
w = FastText.load(r'enron.bin')
all_stopwords = gensim.parsing.preprocessing.STOPWORDS


def preprocess(b):
    lit = re.sub(r'\b[0-9]+\b', ' ', b)
    lit = lit.replace('www.', '').replace('.com', '').replace('inc.', '').replace('.org', '').replace(
        'http', '').replace('.jpg', '').replace('bcc', '').replace('.txt', '').replace('.mpg', '')
    lit = lit.replace('.xls', '').replace('etc.', '').replace('.doc', '').replace(
        '.exe', '').replace('iii', '').replace("“", '').replace("”", '').replace("‚", ' ')

    lit = re.sub(r'[^A-Za-z0-9]+', ' ', lit)
    lit = lit.lower()
    # print(lit)
    shortword = re.compile(r'\W*\b\w{1,3}\b')
    lit = shortword.sub('', lit)
    lit = lit.strip()
    text_tokens = word_tokenize(lit)
    tokenized_sents = [
        word for word in text_tokens if not word in all_stopwords]
    # length = len(tokenized_sents)
    lit = TreebankWordDetokenizer().detokenize(tokenized_sents)
    return lit


def plot_top_words(model, feature_names, n_top_words, title):
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
    return top_features


'''
for the expanded words, provide query like this....
{
    "query": ".......",   (this will be a paragraph)
    "topics": .....,      (this will be a number)
    "correlated": ......  (this will be a number)
}

output query will be like this....
{
    "query": "......",
    "topics": ["....", "....", "...." ....],
    "words": {
        "word1": ["....", "....", "...." ....],
        "word2": ["....", "....", "...." ....],
        .
        .
        .
    }
}
'''

# create API call


@cross_origin()
@app.route('/')
def hello():
    return "API program URL test successful"


@cross_origin()
@app.route('/api/queries/words/', methods=['POST'])
def create_query_expanded_words():
    corpus = request.json['query']
    preprocessed_text = []
    preprocessed_text.append(preprocess(corpus))

    def conv(i): return i or ''
    res = [conv(i) for i in preprocessed_text]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(res)
    LDA = LatentDirichletAllocation(n_components=1, random_state=42)
    LDA.fit(X)

    tf_feature_names = vectorizer.get_feature_names()
    lst2 = plot_top_words(LDA, tf_feature_names,
                          request.json["topics"], 'Topics in LDA model')
    print(lst2)
    # getting the synonyms and correlated words for each topic word in lst2 and storing them in expanded dic
    expanded = defaultdict(list)
    goog = []
    _cmp = []
    count = 0
    for i in lst2:
        syno = []
        corl = []
        try:
            # will try to add the init_sims line here to check if it works.
            goog = w.wv.most_similar(
                positive=[i], topn=request.json['correlated']+len(_cmp))
            list2 = [i[0] for i in goog]
            # print(list2)
            if count > 0:
                l3 = [x for x in list2 if x not in _cmp]
                list2 = l3[:request.json['correlated']]
                # print("list2-> {}".format(len(list2)))
                # print("list2-> {}".format(len(set(list2))))
            for item in list2:
                _cmp.append(item)
                x = item.strip()
                if len(x) > 2:
                    corl.append(x)
                    _cmp.append(x)
        except:
            pass
        for syn in wordnet.synsets(i):
            for l in syn.lemmas():
                x = re.sub('[^a-zA-Z0-9\n]', ' ', l.name()).strip()
                if len(x) > 2 and x not in _cmp:
                    syno.append(x)
                    _cmp.append(x)
        # print("correlated-> {}".format(len(corl)))
        # print("correlated-> {}".format(len(set(corl))))
        if i not in _cmp:
            expanded[i].append(i)
            _cmp.append(i)

        expanded[i].extend(list(OrderedDict.fromkeys(corl)) +
                           list(OrderedDict.fromkeys(syno)))
        expanded[i] = list(OrderedDict.fromkeys(expanded[i]))
        print(len(expanded[i]))
        print(len(set(expanded[i])))
        count += 1
    query_word = {
        'query': request.json['query'],
        'topics': lst2,
        'words': expanded
    }
    return jsonify({'output': query_word}), 201


# main block
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4000, debug=True)
