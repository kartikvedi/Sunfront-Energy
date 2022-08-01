import numpy as np
import pandas as pd

# Extracts the training dataset either as a numpy matrix or as a pandas dataframe
def ReadTrainData(fileName='./data/train.tsv', asNumpy=True):
    df = pd.read_csv(fileName, sep='\t', header=0)
    df.dropna(axis='index', how='any')

    df_features = df.drop(['Sentiment'], axis=1)
    df_targets = df.drop(['PhraseId', 'SentenceId', 'Phrase'], axis=1)

    df_features = df_features.apply(lambda x: x.astype(str).str.lower())
    df_features['PhraseId'] = pd.to_numeric(df_features['PhraseId'])
    df_features['SentenceId'] = pd.to_numeric(df_features['SentenceId'])

    if(asNumpy):
        return df_features.values, np.transpose(df_targets.values)[0]
    else:
        return df_features, df_targets

# Extracts the testing dataset either as a numpy matrix or as a pandas dataframe
def ReadTestData(fileName='./data/test.tsv', asNumpy=True):
    df_features = pd.read_csv(fileName, sep='\t', header=0)
    df_features.dropna(axis='index', how='any')

    df_features = df_features.apply(lambda x: x.astype(str).str.lower())
    df_features['PhraseId'] = pd.to_numeric(df_features['PhraseId'])
    df_features['SentenceId'] = pd.to_numeric(df_features['SentenceId'])

    if(asNumpy):
        return df_features.values
    else:
        return df_features

# Creates a list containing lists of tokens that represent the sentences contained
# in the provided dataset.
def ExtractCorpus(dataset, sentId_col=1, phraseId_col=2):
    corpus = []
    sentIt = -1

    for sent in dataset:
        # UNCOMMENT IF ONLY USING COMPLETE SENTENCES FROM PROVIDED CORPUS
        # if(sent[sentId_col] != sentIt):
        #     corpus.append(sent[phraseId_col].split())
        #     sentIt = sent[sentId_col]

       # This adds lists even if they are not complete sentences.
        corpus.append(sent[phraseId_col].split())

    return corpus
