'''
contains binary sentiment classification models for IMDB dataset
usage: python3 IMDB_models.py --name gru --mode train
1. Linear
2. Transformer
3. GRU
4. LSTM
'''

from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from datasets import load_dataset

tr = load_dataset("imdb", split="train")
te = load_dataset("imdb", split="test")
nominal_train = tr["text"]
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(nominal_train)

class CustomMultiHeadAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=2, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads

        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, heads, seq_len, dim)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention_output = self.attention(query, key, value)

        # Combine heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads
        })
        return config

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = CustomMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })
        return config

def Transformer(maxlen, vocab_size):
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = tf.keras.layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(20, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def GRU(vocab_size, max_length):
    embedding_dim = 32  # Embedding size for each token
    # GRU: model initialization
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
        tf.keras.layers.Dense(24, activation='relu'),
        # tf.keras.layers.Dense(1, activation='sigmoid')
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    # compile model
    # model.compile(loss='binary_crossentropy',
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # model summary
    model.summary()
    return model


def LSTM(vocab_size, max_length):
    embedding_dim = 32  # Embedding size for each token
    # LSTM, model initialization
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(24, activation='relu'),
        # tf.keras.layers.Dense(1, activation='sigmoid')
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    # compile model
    # model.compile(loss='binary_crossentropy',
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # model summary
    model.summary()
    return model

def Linear(vocab_size, maxlen):
    embedding_dim = 32
    model = tf.keras.Sequential([
        # input (bs, 200), output (bs, 200, 32)
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
        tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.Dense(1, activation='sigmoid')
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    # compile model
    # model.compile(loss='binary_crossentropy',
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # model summary
    model.summary()
    return model

TransformerPath = 'model/transformer_tokenized.h5'
LSTMPath = 'model/lstm_tokenized.h5'
GRUPath = 'model/gru_tokenized.h5'
LinearPath = 'model/linear_tokenized.h5'

def main(name, mode='train'):

    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    # (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
    # x_train = pad_sequences(x_train, maxlen=maxlen)
    # x_test = pad_sequences(x_test, maxlen=maxlen)

    # read from data/tokenized instead
    x_train = np.load('data/tokenized/trainX.npy')
    y_train = np.load('data/tokenized/trainy.npy')
    x_test = np.load('data/tokenized/testX.npy')
    y_test = np.load('data/tokenized/testy.npy')

    if name == 'transformer':
        model = Transformer(maxlen, vocab_size)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        if mode == 'train':
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=TransformerPath,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            )
            history = model.fit(
                x_train, y_train, batch_size=16, epochs=5, verbose=2, validation_split=0.3,
                callbacks=[checkpoint_cb]
            )

        if mode == 'test':
            best_model = load_model(TransformerPath, custom_objects={
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": TransformerBlock,
        "CustomMultiHeadAttention": CustomMultiHeadAttention
    })
            # predict values
            pred = best_model.predict(x_test)
            # Take the class with the highest probability
            pred_labels = np.argmax(pred, axis=1)

            # Compute accuracy
            acc = accuracy_score(y_test, pred_labels)
            print(f"Accuracy: {acc:.4f}")


    if name == 'lstm':
        model = LSTM(vocab_size, maxlen)

        if mode == 'train':
            # fit model
            num_epochs = 10
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=LSTMPath,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            )

            history = model.fit(x_train, y_train,
                                epochs=num_epochs, verbose=2, callbacks=[checkpoint_cb],
                                validation_split=0.3)

        if mode == 'test':
            best_model = load_model(LSTMPath)
            # predict values
            pred = best_model.predict(x_test)
            # Convert probabilities to binary predictions (0 or 1)
            pred_labels = (pred > 0.5).astype("int32")

            # Compute accuracy
            acc = accuracy_score(y_test, pred_labels)
            print(f"Accuracy: {acc:.4f}")

    if name =='gru':
        model = GRU(vocab_size, maxlen)
        if mode == 'train':
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=GRUPath,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            )

            # fit model
            num_epochs = 5
            history = model.fit(x_train, y_train,
                                epochs=num_epochs, verbose=2, callbacks=[checkpoint_cb],
                                validation_split=0.3)

        if mode == 'test':
            best_model = load_model(GRUPath)
            # predict values
            pred = best_model.predict(x_test)
            # Convert probabilities to binary predictions (0 or 1)
            pred_labels = (pred > 0.5).astype("int32")

            # Compute accuracy
            acc = accuracy_score(y_test, pred_labels)
            print(f"Accuracy: {acc:.4f}")

    if name == 'linear':
        model = Linear(vocab_size, maxlen)
        if mode == 'train':
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=LinearPath,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            )

            # fit model
            num_epochs = 5
            history = model.fit(x_train, y_train,
                                epochs=num_epochs, verbose=2, callbacks=[checkpoint_cb],
                                validation_split=0.3)

        if mode == 'test':
            best_model = load_model(LinearPath)
            # predict values
            pred = best_model.predict(x_test)
            # Convert probabilities to binary predictions (0 or 1)
            pred_labels = (pred > 0.5).astype("int32")

            # Compute accuracy
            acc = accuracy_score(y_test, pred_labels)
            print(f"Accuracy: {acc:.4f}")

def lm(name, mode=None):
    path=''
    if mode == 'tokenized':
        if name == 'transformer':
            model = load_model(path+'model/transformer_tokenized.h5', custom_objects={
                "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
                "TransformerBlock": TransformerBlock,
                "CustomMultiHeadAttention": CustomMultiHeadAttention
            })
        else:
            model = load_model(path+f'model/{name}_tokenized.h5')
    elif name == 'transformer':
        model = load_model("transformer_imdb.h5", custom_objects={
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "TransformerBlock": TransformerBlock,
            "CustomMultiHeadAttention": CustomMultiHeadAttention
        })
    elif name == 'transformer_tokenizer':
        model = load_model("transformer_tokenizer.h5", custom_objects={
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "TransformerBlock": TransformerBlock,
            "CustomMultiHeadAttention": CustomMultiHeadAttention
        })
    elif name == 'lstm':
        model = load_model("lstm_imdb.h5")
    elif name == 'gru':
        model = load_model("gru_imdb.h5")
    elif name == 'linear':
        model = load_model("linear_imdb.h5")
    return model

# def label_shift():
#     transformer = lm('transformer')
#     (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=20000)
#     np.random.seed(42)

#     N = len(x_test)
#     positiveId = np.where(y_test == 1)[0]
#     negativeId = np.where(y_test == 0)[0]
#     positiveTarget = int(0.8 * N)
#     negativeTarget = N - positiveTarget

#     integerTimes = int(positiveTarget / len(positiveId))
#     selectedId = []
#     for i in range(integerTimes):
#         selectedId.extend(positiveId)
#     remainder = positiveTarget - len(selectedId)
#     if remainder > 0:
#         selectedId.extend(np.random.choice(positiveId, remainder, replace=False))
#     selectedId.extend(np.random.choice(negativeId, negativeTarget, replace=False))

#     selectedX, selectedY = x_test[selectedId], y_test[selectedId]
#     selectedX = pad_sequences(selectedX, maxlen=200)

#     if not os.path.exists('data/label'):
#         os.makedirs('data/label')
#     np.save('data/label/X.npy', selectedX)
#     np.save('data/label/y.npy', selectedY)

#     pred = transformer.predict(selectedX)
#     pred_labels = np.argmax(pred, axis=1)
#     acc = accuracy_score(selectedY, pred_labels)
#     print(f"Accuracy: {acc:.4f}")

def covariate_prepare_corrupted(severity=0.5):
    import corrupted_text
    from datasets import load_dataset

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

    # np.random.seed(42)
    # N = len(nominal_test)
    # ids = np.arange(N)
    # np.random.shuffle(ids)
    # corruptedId = ids[:N // 2]
    # pristineId = ids[N // 2:]
    # y_test = np.array(te["label"])
    # selectedX = np.concatenate((x_test_corrupted[corruptedId], x_test[pristineId]), axis=0)
    # selectedY = np.concatenate((y_test[corruptedId], y_test[pristineId]), axis=0)
    # np.save('data/covariate/X.npy', selectedX)
    # np.save('data/covariate/y.npy', selectedY)

def covariate_prepare_transformer():
    from datasets import load_dataset

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

    # ValueError: `validation_split` is only supported for Tensors or NumPy arrays, found:
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
    from datasets import load_dataset

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
    from datasets import load_dataset

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

def test_transformer_tokenizer():
    model = lm('lstm', mode='tokenized')
    x_test = np.load('data/tokenized/testX.npy')
    y_test = np.load('data/tokenized/testy.npy')
    pred = model.predict(x_test)
    print("pred.shape", pred.shape)
    print("pred.head", pred[:5])
    pred_labels = np.argmax(pred, axis=1)
    # pred_labels = (pred > 0.5).astype(int)
    acc = accuracy_score(y_test, pred_labels)
    print(f"Accuracy: {acc:.4f}")

# covariate_prepare_corrupted()
# covariate_prepare_transformer()
# covariate_shift() # 0.7499
# natural_covariate_prepare_dataset()
# natural_covariate_shift() # 0.7890
# adversarial_shift()
# label_shift()
# label_shift_tokenizer()
# prepare_tokenized_dataset()
# main(name='transformer', mode='train')
# main(name='lstm', mode='train')
# main(name='gru', mode='train')
# main(name='linear', mode='train')
# test_transformer_tokenizer()

from textattack.models.wrappers import ModelWrapper
class CustomTensorFlowModelWrapper(ModelWrapper):
    def __init__(self, model):
        self.model = model
    def __call__(self, text_input_list):
        trainX = preprocess_input(text_input_list)
        final_preds = self.model.predict(trainX)
        return final_preds

def preprocess_input(text_input_list):
    X = tokenizer.texts_to_sequences(text_input_list)
    X = pad_sequences(X, maxlen=200)
    return X

from textattack.transformations import WordSwap

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
    from textattack.search_methods import GreedySearch
    from textattack.constraints.pre_transformation import (
        RepeatModification,
        StopwordModification,
    )
    from textattack import Attack
    # Create the goal function using the model
    from textattack.goal_functions import UntargetedClassification

    goal_function = UntargetedClassification(model_wrapper)
    # We're going to use our Banana word swap class as the attack transformation.
    transformation = BananaWordSwap()
    # We'll constrain modification of already modified indices and stopwords
    constraints = [RepeatModification(), StopwordModification()]
    # We'll use the Greedy search method
    search_method = GreedySearch()
    return Attack(goal_function, constraints, transformation, search_method)

def create_adv():
    from IMDB_models import lm
    from textattack.attack_recipes import PWWSRen2019
    import random
    from tqdm import tqdm
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
    covariate_prepare_corrupted(1.0)

    '''
    transformer = lm('transformer_tokenizer')
    model_wrapper = CustomTensorFlowModelWrapper(transformer)

    #If an example is incorrectly predicted to begin with, it is not attacked (SKIPPED), since the input already fools the model.
    for i in range(len(te)):
        pred = transformer.predict(preprocess_input([te[i]['text']]))
        #print('original probabiility', pred, 'processed input', preprocess_input([te[i]['text']]))
        print('ground truth label', te[i]['label'], 'original prediction label', np.argmax(pred, axis=1), pred, 'sample', i)
        result = attack.attack(te[i]['text'], te[i]['label'])
        print('attacked sample:', result)
        processed_vector = preprocess_input([result.perturbed_result.attacked_text.printable_text()])
        breakpoint()
    '''