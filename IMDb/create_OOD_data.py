from textattack.transformations import WordSwap
from textattack.search_methods import GreedySearch
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack import Attack
from textattack.goal_functions import UntargetedClassification
from IMDB_models import lm, Transformer, tr, te, nominal_train, tokenizer
from textattack.attack_recipes import PWWSRen2019
import random
from tqdm import tqdm
from textattack.models.wrappers import ModelWrapper
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.metrics import accuracy_score
import os
import corrupted_text
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf

def label_shift():
    transformer = lm('transformer')
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=20000)
    np.random.seed(42)

    N = len(x_test)
    positiveId = np.where(y_test == 1)[0]
    negativeId = np.where(y_test == 0)[0]
    positiveTarget = int(0.8 * N)
    negativeTarget = N - positiveTarget

    integerTimes = int(positiveTarget / len(positiveId))
    selectedId = []
    for i in range(integerTimes):
        selectedId.extend(positiveId)
    remainder = positiveTarget - len(selectedId)
    if remainder > 0:
        selectedId.extend(np.random.choice(positiveId, remainder, replace=False))
    selectedId.extend(np.random.choice(negativeId, negativeTarget, replace=False))

    selectedX, selectedY = x_test[selectedId], y_test[selectedId]
    selectedX = pad_sequences(selectedX, maxlen=200)

    if not os.path.exists('data/label'):
        os.makedirs('data/label')
    np.save('data/label/X.npy', selectedX)
    np.save('data/label/y.npy', selectedY)

    pred = transformer.predict(selectedX)
    pred_labels = np.argmax(pred, axis=1)
    acc = accuracy_score(selectedY, pred_labels)
    print(f"Accuracy: {acc:.4f}")


def covariate_prepare_corrupted(severity=0.5):
    vocab_size = 20000
    maxlen = 200

    tr = load_dataset("imdb", split="train")
    te = load_dataset("imdb", split="test")
    nominal_train = tr["text"]
    nominal_test = te["text"]

    corruptor = corrupted_text.TextCorruptor(base_dataset=nominal_test + nominal_train, cache_dir=".mycache")
    imdb_corrupted = corruptor.corrupt(nominal_test, severity=severity, seed=1)
    if not os.path.exists('data/covariate'):
        os.makedirs('data/covariate')

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(nominal_train)
    x_test = tokenizer.texts_to_sequences(nominal_test)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    x_test_corrupted = tokenizer.texts_to_sequences(imdb_corrupted)
    x_test_corrupted = pad_sequences(x_test_corrupted, maxlen=maxlen)
    np.save(f'data/covariate/X_corrupted_{severity}.npy', x_test_corrupted)

    np.random.seed(42)
    N = len(nominal_test)
    ids = np.arange(N)
    np.random.shuffle(ids)
    corruptedId = ids[:N // 2]
    pristineId = ids[N // 2:]
    y_test = np.array(te["label"])
    selectedX = np.concatenate((x_test_corrupted[corruptedId], x_test[pristineId]), axis=0)
    selectedY = np.concatenate((y_test[corruptedId], y_test[pristineId]), axis=0)
    np.save('data/covariate/X.npy', selectedX)
    np.save('data/covariate/y.npy', selectedY)


def covariate_prepare_transformer():
    vocab_size = 20000
    maxlen = 200
    tr = load_dataset("imdb", split="train")
    nominal_train = tr["text"]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(nominal_train)
    trainX = tokenizer.texts_to_sequences(nominal_train)
    trainX = pad_sequences(trainX, maxlen=maxlen)
    trainy = tr["label"]

    model = Transformer(maxlen, vocab_size)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    trainX = tf.convert_to_tensor(trainX)
    trainy = tf.convert_to_tensor(trainy)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath='transformer_tokenizer.h5',
        save_best_only=True,  # saves only the best model (based on val_loss)
        monitor='val_loss',  # metric to monitor
        mode='min',  # minimize val_loss
        verbose=1
    )
    history = model.fit(
        trainX, trainy, batch_size=16, epochs=5, verbose=2, validation_split=0.3,
        callbacks=[checkpoint_cb]
    )


def covariate_shift():
    X = np.load('data/covariate/X.npy')
    y = np.load('data/covariate/y.npy')

    transformer = lm('transformer_tokenizer')
    pred = transformer.predict(X)
    pred_labels = np.argmax(pred, axis=1)
    acc = accuracy_score(y, pred_labels)
    print(f"Accuracy: {acc:.4f}")


def natural_covariate_prepare_dataset():
    tr = load_dataset("imdb", split="train")
    te = load_dataset("imdb", split="test")
    nominal_train = tr["text"]
    nominal_test = te["text"]
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(nominal_train)

    # format: text + \t + label, last line is empty
    covariateText = []
    covariateLabel = []
    files = ["data/customer/CR.train", "data/customer/CR.test", "data/customer/CR.dev"]

    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line == "\n":
                    continue
                covariateText.append(line.split("\t")[0])
                covariateLabel.append(int(line.split("\t")[1].strip()))
    x_test = tokenizer.texts_to_sequences(nominal_test)
    x_test = pad_sequences(x_test, maxlen=200)
    x_covariate = tokenizer.texts_to_sequences(covariateText)
    x_covariate = pad_sequences(x_covariate, maxlen=200)
    y_test = np.array(te["label"])
    y_covariate = np.array(covariateLabel)

    np.random.seed(42)
    N = len(x_covariate)
    print("test size: ", N * 2)
    # select that much from x_test
    selectedId = np.random.choice(N, N, replace=False)
    selectedX = np.concatenate((x_test[selectedId], x_covariate), axis=0)
    selectedY = np.concatenate((y_test[selectedId], y_covariate), axis=0)
    np.save('data/customer/X.npy', selectedX)
    np.save('data/customer/y.npy', selectedY)


def natural_covariate_shift():
    transformer = lm('transformer_tokenizer')
    x_test = np.load('data/customer/X.npy')
    y_test = np.load('data/customer/y.npy')
    pred = transformer.predict(x_test)
    pred_labels = np.argmax(pred, axis=1)
    acc = accuracy_score(y_test, pred_labels)
    print(f"Accuracy: {acc:.4f}")


def label_shift_tokenizer():
    transformer = lm('transformer_tokenizer')
    tr = load_dataset("imdb", split="train")
    te = load_dataset("imdb", split="test")
    nominal_train = tr["text"]
    nominal_test = te["text"]
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(nominal_train)
    testX = tokenizer.texts_to_sequences(nominal_test)
    testX = pad_sequences(testX, maxlen=200)
    testy = np.array(te["label"])

    np.random.seed(42)
    N = len(testX)
    positiveId = np.where(testy == 1)[0]
    negativeId = np.where(testy == 0)[0]
    positiveTarget = int(0.8 * N)
    negativeTarget = N - positiveTarget
    integerTimes = int(positiveTarget / len(positiveId))
    selectedId = []
    for i in range(integerTimes):
        selectedId.extend(positiveId)
    remainder = positiveTarget - len(selectedId)
    if remainder > 0:
        selectedId.extend(np.random.choice(positiveId, remainder, replace=False))
    selectedId.extend(np.random.choice(negativeId, negativeTarget, replace=False))

    selectedX, selectedy = testX[selectedId], testy[selectedId]

    if not os.path.exists('data/label'):
        os.makedirs('data/label')
    np.save('data/label/X.npy', selectedX)
    np.save('data/label/y.npy', selectedy)

    pred = transformer.predict(selectedX)
    pred_labels = np.argmax(pred, axis=1)
    acc = accuracy_score(selectedy, pred_labels)
    print(f"Accuracy: {acc:.4f}")


def prepare_tokenized_dataset():
    vocab_size = 20000
    maxlen = 200

    tr = load_dataset("imdb", split="train")
    te = load_dataset("imdb", split="test")
    nominal_train = tr["text"]
    nominal_test = te["text"]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(nominal_train)
    trainX = tokenizer.texts_to_sequences(nominal_train)
    trainX = pad_sequences(trainX, maxlen=maxlen)
    trainy = tr["label"]
    testX = tokenizer.texts_to_sequences(nominal_test)
    testX = pad_sequences(testX, maxlen=maxlen)
    testy = te["label"]
    if not os.path.exists('data/tokenized'):
        os.makedirs('data/tokenized')
    np.save('data/tokenized/trainX.npy', trainX)
    np.save('data/tokenized/trainy.npy', trainy)
    np.save('data/tokenized/testX.npy', testX)
    np.save('data/tokenized/testy.npy', testy)

class CustomTensorFlowModelWrapper(ModelWrapper):
    def __init__(self, model):
        self.model = model
    def __call__(self, text_input_list):
        trainX = preprocess_input(text_input_list)
        final_preds = self.model.predict(trainX)
        return final_preds
class BananaWordSwap(WordSwap):
    """Transforms an input by replacing any word with 'banana'."""

    # We don't need a constructor, since our class doesn't require any parameters.

    def _get_replacement_words(self, word):
        """Returns 'banana', no matter what 'word' was originally.

        Returns a list with one item, since `_get_replacement_words` is intended to
            return a list of candidate replacement words.
        """
        return ["banana"]
def BananaSwap():
    goal_function = UntargetedClassification(model_wrapper)
    # We're going to use our Banana word swap class as the attack transformation.
    transformation = BananaWordSwap()
    # We'll constrain modification of already modified indices and stopwords
    constraints = [RepeatModification(), StopwordModification()]
    # We'll use the Greedy search method
    search_method = GreedySearch()
    return Attack(goal_function, constraints, transformation, search_method)

def preprocess_input(text_input_list):
    X = tokenizer.texts_to_sequences(text_input_list)
    X = pad_sequences(X, maxlen=200)
    return X

def test_transformer_tokenizer():
    model = lm('lstm', mode='tokenized')
    x_test = np.load('data/tokenized/testX.npy')
    y_test = np.load('data/tokenized/testy.npy')
    pred = model.predict(x_test)
    pred_labels = np.argmax(pred, axis=1)
    # pred_labels = (pred > 0.5).astype(int)
    acc = accuracy_score(y_test, pred_labels)

def create_adv():
    transformer = lm('transformer_tokenizer')
    model_wrapper = CustomTensorFlowModelWrapper(transformer)
    attacks = [
        PWWSRen2019.build(model_wrapper),
        BananaSwap()
    ]
    # Identify correctly predicted test samples
    pathX, pathy = './data/tokenized/testX.npy', './data/tokenized/testy.npy'
    processed_testX, labels = np.load(pathX), np.load(pathy)
    correct_indices = np.load('./correct_indices.npy')
    # Randomly select 12500 correctly predicted samples
    selected_indices = np.random.choice(correct_indices, size=12500, replace=False)

    # Store final data
    X_final = []
    y_final = []

    # Add 50% original samples
    original_indices = np.random.choice(np.arange(len(te)), size=12500, replace=False)

    # Add 50% perturbed samples using attacks
    random.shuffle(selected_indices)  # Shuffle for attack diversity
    for idx, i in tqdm(enumerate(selected_indices)):
        print('chosen attack method', attacks[idx % 2], 'idx', idx % 2)
        i = int(i)
        attack = attacks[idx % 2]
        text = te[i]['text']
        label = te[i]['label']
        try:
            result = attack.attack(text, label)
            if result.perturbed_result is None:
                continue
            perturbed_text = result.perturbed_result.attacked_text.printable_text()
            vec = preprocess_input([perturbed_text])[0]
            X_final.append(vec)
            y_final.append(label)
        except Exception as e:
            print(f"Attack failed on sample {i}: {e}")
            continue
        if len(X_final) >= 25000:
            break

    # Save to .npy files
    X_final = np.vstack((processed_testX[original_indices], np.array(X_final)))
    y_final = np.hstack((labels[original_indices], np.array(y_final)))
    np.save("X_attacked_clean.npy", X_final)
    np.save("y.npy", y_final)

    print("Saved X.npy with shape", X_final.shape)
    print("Saved y.npy with shape", y_final.shape)

if __name__ == '__main__':
    prepare_tokenized_dataset()
    covariate_prepare_corrupted()
    covariate_prepare_transformer()
    covariate_shift() # 0.7499
    natural_covariate_prepare_dataset()
    natural_covariate_shift() # 0.7890
    label_shift()
    label_shift_tokenizer()

    # create adversarial text data, run the below:
    test_transformer_tokenizer()
    transformer = lm('transformer_tokenizer')
    model_wrapper = CustomTensorFlowModelWrapper(transformer)
    # If an example is incorrectly predicted to begin with, it is not attacked (SKIPPED), since the input already fools the model.
    for i in range(len(te)):
        pred = transformer.predict(preprocess_input([te[i]['text']]))
        # print('original probabiility', pred, 'processed input', preprocess_input([te[i]['text']]))
        print('ground truth label', te[i]['label'], 'original prediction label', np.argmax(pred, axis=1), pred,
              'sample', i)
        result = attack.attack(te[i]['text'], te[i]['label'])
        print('attacked sample:', result)
        processed_vector = preprocess_input([result.perturbed_result.attacked_text.printable_text()])