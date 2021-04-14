from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from extract_irony_sentiment_feature import parse_trainset
import pandas as pd
import numpy as np
import tqdm

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

sentences, labels = parse_trainset("/Users/sylvia/PycharmProjects/IR_project/irony_dataset/SemEval2018-T3-train-taskA.txt")
processed_sentences = []
for s in sentences:

    processed_sentences.append(" ".join(text_processor.pre_process_doc(s)))

processed_df = pd.DataFrame({"text":processed_sentences, "label":labels})
processed_df.to_csv("/Users/sylvia/PycharmProjects/IR_project/irony_dataset/with_space/preprocessd_train.csv", index=None)
# processed_df_train = pd.read_csv("/Users/sylvia/PycharmProjects/IR_project/irony_dataset/preprocess_df.csv", index_col=None)

def word_count(df):
    import collections
    word_freq = collections.defaultdict(int)
    for i in range(len(df)):
        for j in df.loc[i, "text"].strip().split(","):
            word_freq[j] += 1
    return word_freq


def build_dict(file_name, min_word_freq=5):
    word_freq = word_count(file_name)
    word_freq = filter(lambda x: x[1] > min_word_freq, word_freq.items())
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(zip(words, range(len(words))))
    word_idx['<unk>'] = len(words) #unk表示unknown，未知单词
    return word_idx


glove_file = "pretrained_embedding/glove.twitter.27B.200d.txt"

EMBEDDING_VECTOR_LENGTH = 200
def construct_embedding_matrix(glove_file, word_index):
    embedding_dict = {}
    with open(glove_file,'r') as f:
        for line in f:
            values=line.split()
            # get the word
            word=values[0]
            if word in word_index.keys():
                # get the vector
                vector = np.asarray(values[1:], 'float32')
                embedding_dict[word] = vector
    ###  oov words (out of vacabulary words) will be mapped to 0 vectors

    num_words=len(word_index)+1
    #initialize it to 0
    embedding_matrix=np.zeros((num_words, EMBEDDING_VECTOR_LENGTH))

    for word, i in tqdm.tqdm(word_index.items()):
        if i < num_words:
            vect=embedding_dict.get(word, [])
            if len(vect)>0:
                embedding_matrix[i] = vect[:EMBEDDING_VECTOR_LENGTH]
    return embedding_matrix
# embedding_matrix =  construct_embedding_matrix(glove_file, build_dict(processed_df, 5))
# np.save("glove_twitter_embed_matrix.npy", embedding_matrix)

# processed_df["token_to_idx"] = 0
def embed_index(df, word_index):
    for i in range(len(df)):
        index_list = []
        for word in df.loc[i, "text"].strip().split(","):
            if word in word_index.keys():
                index_list.append(word_index[word])
        df.loc[i, 'token_to_idx'] = str(index_list)
    return df
embed_index(processed_df, build_dict(processed_df_train, 5)).to_csv("/Users/sylvia/PycharmProjects/IR_project/irony_dataset/token_idx_test.csv", index=None)






