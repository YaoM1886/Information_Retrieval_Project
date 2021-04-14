from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import csv


pd.set_option("max_columns", None)


def parse_trainset(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with open(fp, 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)

    return corpus, y


def sentiment_analyzer(corpus):
    analyzer = SentimentIntensityAnalyzer()
    pos_ratio = []
    neg_ratio = []
    neu_ratio = []
    sentiment_score = []
    for sentence in corpus:
        vs = analyzer.polarity_scores(sentence)
        pos_ratio.append(vs["pos"])
        neg_ratio.append(vs["neg"])
        neu_ratio.append(vs["neu"])
        sentiment_score.append(vs["compound"])

    sentiment_df = pd.DataFrame({"corpus":corpus,
                                 "pos_ratio":pos_ratio,
                                 "neg_ratio":neg_ratio,
                                 "neu_ratio":neu_ratio,
                                 "sentiment_score":sentiment_score})

    return sentiment_df


def to_arff_file(X_train, y_train):
    csv_train = pd.DataFrame({"Tweet index": range(len(X_train)), "Tweet text": X_train, "Label": y_train})
    csv_train["Tweet text"] = csv_train["Tweet text"].map(lambda x: str(x).replace('"', '').strip())
    csv_train["Tweet text"] = csv_train["Tweet text"].map(lambda x: f'"{x}"')
    csv_train.to_csv("/Users/sylvia/PycharmProjects/IR_project/irony_dataset/test_taskA_csv.csv", escapechar='\\',
                     quoting=csv.QUOTE_NONE, index=None)

def read_arff(path):
    df = pd.read_csv(path, index_col=None, header=None)
    return df

if __name__ == "__main__":
    X_train, y_train = parse_trainset("/Users/sylvia/PycharmProjects/IR_project/irony_dataset/SemEval2018-T3-train-taskA.txt")
    print("Train set size:", len(X_train))

    X_test, y_test = parse_trainset("/Users/sylvia/PycharmProjects/IR_project/irony_dataset/SemEval2018-T3_input_test_taskA_labeled.txt")
    print("Test set size:", len(X_test))

    # train arff
    train_arff = read_arff("/Users/sylvia/Desktop/irony_sentiment.csv")
    # train vader
    train_vader = sentiment_analyzer(X_train)
    train_sent = pd.concat([train_arff, train_vader], axis=1)
    train_sent.drop(["corpus", 0], inplace=True, axis=1)
    print(train_sent.shape)
    train_sent.to_csv("/Users/sylvia/Desktop/train_sentiment.csv", index=None, header=0)


    # test arff
    test_arff = read_arff("/Users/sylvia/Desktop/test_arff.csv")
    # test vader
    test_vader = sentiment_analyzer(X_test)
    test_sent = pd.concat([test_arff, test_vader], axis=1)
    test_sent.drop(["corpus", 0], inplace=True, axis=1)
    test_sent.to_csv("/Users/sylvia/Desktop/test_sentiment.csv", index=None, header=0)







    #






