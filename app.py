from flask import Flask, render_template, request, redirect

def funct(text):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from scipy.sparse import hstack
    from sklearn.model_selection import train_test_split

    gr = pd.read_csv('gossipcop_real.csv')
    pr = pd.read_csv('politifact_real.csv')
    gf = pd.read_csv('gossipcop_fake.csv')
    pf = pd.read_csv('politifact_fake.csv')
    fr = pd.read_csv('truth1.csv')

    real = pd.concat([gr, pr, fr], sort=False)
    real['fake'] = 0
    real = real[['title', 'fake']]

    fake = pd.concat([gf, pf])
    fake['fake'] = 1
    fake = fake[['title', 'fake']]

    df = pd.concat([real, fake])
    df.dropna(inplace=True)

    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 1),
        max_features=10000)
    train = word_vectorizer.fit_transform(df['title'])
    target = df['fake']

    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=0.3)

    classifier = LogisticRegression(C=0.1, solver='sag')
    classifier.fit(X_train, y_train)

    l = classifier.predict(X_test)

    from sklearn.metrics import roc_auc_score
    roc_auc_score(l, y_test)
    
    ts = word_vectorizer.transform([text])
    prediction = classifier.predict(ts)
    return prediction

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        x = request.form['content']
        prediction = funct(x)
        return render_template('a.html',sent=prediction)
    else:
        return render_template('a.html')

if __name__ == "__main__":
    app.run(debug=True)
