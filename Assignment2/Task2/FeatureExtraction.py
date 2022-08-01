import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from Util import ReadTrainData, ReadTestData, ExtractCorpus

# CONSTANTS
MODEL_DIR = "./models/word2vec_corpus.model"
# Uncomment if using KeyedVectors pre-trained models - https://nlp.stanford.edu/projects/glove/
# MODEL_DIR = "./models/glove.6B.100d.model"
# GLOVE_DIR = "./models/glove.6B.100d.txt"
TRAIN_DIR = "./data/train_features.csv"
TEST_DIR  = "./data/test_features.csv"
TARG_DIR  = "./data/targets.csv"
RETRAIN   = True
VEC_SIZE  = 100

# Given a list of tokens (sentence), returns the average word vector encoding
# of the tokens contained.
def GenerateFeatVector(sentence, vec_size = VEC_SIZE, model=None):
    if(model is None):
        model = Word2Vec.load(MODEL_DIR)
    
    featVector = np.empty((0, vec_size))

    for word in sentence:
        featVector = np.append(featVector, [model.wv[word]], axis=0)

    return np.average(featVector, axis=0).reshape(1, vec_size)

# Given a list of lists of tokens (sentences), returns the feature vectors
# to be used as examples.
def GenerateFeatMatrix(sentences, vec_size = VEC_SIZE, model=None):
    it = 0
    featMatrix = np.empty((sentences.shape[0], vec_size))

    for sentence in sentences:
        sentence = sentence.split()
        featMatrix[it, :] = GenerateFeatVector(sentence, vec_size=vec_size, model=model)
        it += 1

    return featMatrix

def main():
    # Read datasets
    train, targets = ReadTrainData()
    test = ReadTestData()
    
    # Retrain Word2Vec if needed
    if(RETRAIN):
        full_dataset = np.append(train, test, axis=0)
        corpus = ExtractCorpus(full_dataset)
        model = Word2Vec(corpus, size=VEC_SIZE, window=5, min_count=1, workers=4)
        model.train(corpus, total_examples=len(corpus), epochs=10)
        model.save(MODEL_DIR)

    # Load Model
    model = Word2Vec.load(MODEL_DIR)

    # Uncomment if using KeyedVectors pre-trained models - https://nlp.stanford.edu/projects/glove/
    # _ = glove2word2vec(GLOVE_DIR, MODEL_DIR)
    # model = KeyedVectors.load_word2vec_format(MODEL_DIR, binary=False)

    # Check the model was loaded succesfully
    print(model.wv["happy"])

    train_sent = train[:, 2]
    targets    = targets
    test_sent  = test[:, 2]

    train_feat = GenerateFeatMatrix(train_sent, vec_size = VEC_SIZE, model=model)
    print("Finished Training Matrix")
    test_feat = GenerateFeatMatrix(test_sent, vec_size = VEC_SIZE, model=model)
    print("Finished Test Matrix")

    np.savetxt(TRAIN_DIR, train_feat, delimiter=',')
    np.savetxt(TARG_DIR, targets, delimiter=',')
    np.savetxt(TEST_DIR, test_feat, delimiter=',')



if __name__ == "__main__":
    main()
