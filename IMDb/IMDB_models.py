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


if __name__ == '__main__':
    main(name='transformer', mode='train')

    '''
    main(name='lstm', mode='train')
    main(name='gru', mode='train')
    main(name='linear', mode='train')
    '''