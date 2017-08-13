#!/usr/bin/env python
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from nltk.stem.porter import PorterStemmer
from sklearn import svm
import logging


logger = logging.getLogger(__name__)


#HAM_FILES = ["examples/easy_ham/000*"]
#SPAM_FILES = ["examples/spam/000*"]
HAM_FILES = ["examples/easy_ham/*", "examples/easy_ham_2/*", "examples/hard_ham/*"]
SPAM_FILES = ["examples/spam/*", "examples/spam_2/*"]

# Vocabulary list size
#N = 1900
N = 10000

TRAINING_SET_PROPORTION = 0.60
CROSS_VALIDATION_SET_PROPORTION = 0.20
TEST_SET_PROPORTION = 0.20


def unglob(patterns):
    files = []
    for item in patterns:
        files.extend(glob.glob(item))
    return files


def preprocess_email(email):
    # remove headers
    email_contents = " ".join(email.split("\n\n")[1:])

    # convert to lower case
    email_contents = email_contents.lower()

    # strip all HTML
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # handle numbers
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # handle URLs
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # handle email addresses (strings with '@' in the middle)
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # handle $ characters
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    return email_contents


def tokenize_email(email_contents):
    tokenized_email = []
    stemmer = PorterStemmer()

    # tokenize and get rid of punctuation
    tokens = re.split(r'\W+', email_contents.strip())

    for token in tokens:
        # remove any non-alphanumeric characters
        token = re.sub(r'[^a-z0-9]+', '', token)

        # stem the word
        token = stemmer.stem(token)

        # skip if the word is too short
        if len(token) < 1:
            continue

        tokenized_email.append(token)
    return tokenized_email


def frequency_histogram(files):
    h = Counter()
    for file in files:
        with open(file, 'r', encoding='latin-1') as f:
            email = f.read()
        preprocessed_email = preprocess_email(email)
        tokenized_email = tokenize_email(preprocessed_email)
        h.update(Counter(tokenized_email))
    return h


def plot_frequency_histogram(fh, n=20):
    labels, values = zip(*fh.items())
    ind_sort = np.argsort(values)[::-1][:n]
    labels = np.array(labels)[ind_sort]
    values = np.array(values)[ind_sort]

    # normalize
    values_sum = sum(values)
    values = values / values_sum

    indexes = np.arange(len(labels))
    bar_width = 0.0

    fig = plt.figure(figsize=plt.figaspect(0.2))
    ax = fig.add_subplot(111)
    ax.bar(indexes, values)
    ax.set_xticks(indexes + bar_width)
    ax.set_xticklabels(labels, rotation=45, horizontalalignment="right")
    fig.subplots_adjust(bottom=0.25)  # make room for angled labels
    plt.show(block=False)


def process_email(email_contents, vocab_lookup):
    preprocessed_email = preprocess_email(email_contents)
    tokenized_email = tokenize_email(preprocessed_email)

    # look up index of each token in vocab_list
    word_indices = []
    for token in tokenized_email:
        index = vocab_lookup.get(token, -1)
        if index >= 0:
            word_indices.append(index)
    return word_indices


def load_emails(files, vocab_lookup):
    """Map emails to vocab table, returning a mxn matrix of
       email contents mapping to vocab table, where m is the number
       of examples (files) and n is the vocab size.
    """
    m = len(files)
    n = len(vocab_lookup)
    X = np.zeros((m, n))

    for i, file in enumerate(files):
        with open(file, 'r', encoding='latin-1') as f:
            email_contents = f.read()
        word_indices = process_email(email_contents, vocab_lookup)
        x = np.zeros((n, 1))
        x[word_indices, 0] = 1
        X[i, :] = x.T

    return X


def load_all_data(vocab_lookup, ham_files, spam_files):

    X_ham = load_emails(ham_files, vocab_lookup)
    y_ham = np.zeros((X_ham.shape[0], 1))

    X_spam = load_emails(spam_files, vocab_lookup)
    y_spam = np.ones((X_spam.shape[0], 1))

    X = np.concatenate((X_ham, X_spam))
    y = np.concatenate((y_ham, y_spam))
    return X, y


def train(X, y, C=1.0):
    model = svm.LinearSVC(C=C)
    fit_result = model.fit(X, y.ravel())
    logger.debug(fit_result)
    return model


def predict(model, X):
    return model.predict(X).reshape((-1, 1))


def get_top_spam_predictors(model, vocab_list, N):
    top_indices = np.argsort(model.coef_).ravel()[-N:][::-1]
    return [(vocab_list[x], model.coef_[0, x]) for x in top_indices]


def get_top_ham_predictors(model, vocab_list, N):
    top_indices = np.argsort(model.coef_).ravel()[:N]
    return [(vocab_list[x], model.coef_[0, x]) for x in top_indices]


def plot_predictors(top_spam, top_ham):
    top_spam_labels, top_spam_weights = zip(*top_spam)
    top_ham_labels, top_ham_weights = zip(*top_ham[::-1])

    labels = top_spam_labels + top_ham_labels
    weights = top_spam_weights + top_ham_weights

    indexes = np.arange(len(labels))
    bar_width = 0.0

    colors = ['red' if c > 0 else 'green' for c in weights]

    fig = plt.figure(figsize=plt.figaspect(0.2))
    ax = fig.add_subplot(111)
    ax.bar(indexes, weights, color=colors)
    ax.set_xticks(indexes + bar_width)
    ax.set_xticklabels(labels, rotation=45, horizontalalignment="right")
    fig.subplots_adjust(bottom=0.25)  # make room for angled labels
    plt.show(block=False)


def main():
    logging.basicConfig(level=logging.DEBUG)

    # predictable random numbers
    np.random.seed(0)

    # Expand globs
    ham_files = unglob(HAM_FILES)
    spam_files = unglob(SPAM_FILES)

    # Pass 1 - load email/spam, pre-process, tokenize, then construct frequency histogram
    fh = frequency_histogram(ham_files + spam_files)
    plot_frequency_histogram(fh)
    vocab_list = [word for word, word_count in fh.most_common(N)]

    # create an efficient lookup version vocab_list
    vocab_lookup = {x: i for i, x in enumerate(vocab_list)}

    # Pass 2 - load email/spam, pre-process, tokenize, split into training, cross-validation and test sets:
    #  - Use training set to train the SVM.
    #  - Use cross-validation set to determine performance.
    #  - Use test set for final performance measurement.

    # get a [m x N] matrix of all data examples
    X, y = load_all_data(vocab_lookup, ham_files, spam_files)

    m, n = X.shape

    indices_shuffled = np.arange(m)
    np.random.shuffle(indices_shuffled)

    m_train = int(np.floor(m * TRAINING_SET_PROPORTION))
    m_xvalid = int(np.floor(m * CROSS_VALIDATION_SET_PROPORTION))
    m_test = m - (m_train + m_xvalid)

    X_train = X[indices_shuffled[0:m_train], :]
    y_train = y[indices_shuffled[0:m_train], :]

    X_xvalid = X[indices_shuffled[m_train:m_train+m_xvalid], :]
    y_xvalid = y[indices_shuffled[m_train:m_train+m_xvalid], :]

    X_test = X[indices_shuffled[m_train+m_xvalid:], :]
    y_test = y[indices_shuffled[m_train+m_xvalid:], :]

    # C can be thought of as similar to 1/lambda - therefore increasing C
    # will move towards over-fitting (high variance) and decreasing C will
    # move towards under-fitting (high bias).
    C = 0.03
    model = train(X_train, y_train, C)
    print(f"Model trained with {m_train} examples")

    # top spam predictors
    top_spam_predictors = get_top_spam_predictors(model, vocab_list, 20)
    print("Top spam predictors:")
    for token, weight in top_spam_predictors:
        print(f"  {token:<20} : {weight:.4f}")
    print()

    # top ham predictors
    top_ham_predictors = get_top_ham_predictors(model, vocab_list, 20)
    print("Top ham predictors:")
    for token, weight in top_ham_predictors:
        print(f"  {token:<20} : {weight:.4f}")
    print()

    plot_predictors(top_spam_predictors, top_ham_predictors)

    # Training accuracy
    y_train_predict = predict(model, X_train)
    accuracy_train = np.mean(np.double(y_train_predict == y_train))
    print(f"Training accuracy ({m_train} examples): {accuracy_train * 100:.2f} %")

    # Cross-validation accuracy
    y_xvalid_predict = predict(model, X_xvalid)
    accuracy_xvalid = np.mean(np.double(y_xvalid_predict == y_xvalid))
    print(f"Cross Validation accuracy ({m_xvalid} examples): {accuracy_xvalid * 100:.2f} %")

    # Test set accuracy
    y_test_predict = predict(model, X_test)
    accuracy_test = np.mean(np.double(y_test_predict == y_test))
    print(f"Test accuracy ({m_test} examples): {accuracy_test * 100:.2f} %")

    plt.show()


if __name__ == '__main__':
    main()
