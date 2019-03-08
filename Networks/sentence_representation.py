from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.layers import Input, Dense, Masking, TimeDistributed, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam
import os


class S2SModel(object):
    def __init__(self, config, language):
        self.config = config
        self.language = language
        self.build()

    def build(self):

        word2vec_embedding_dim = self.config["params"]["word2vec"]["embedding_dim"]

        embedding_dim = self.config["params"]["sentence_representation"]["embedding_dim"]
        max_sentence_length = self.config["params"]["sentence_representation"]["max_sentence_length"]
        pad_char_index = self.config["params"]["sentence_representation"]["pad_char_index"]
        learning_rate = self.config["hyper_params"]["sentence_representation"]["lr"]

        word_encoder_model, word_decoder_model = self.get_embedding_model()
        for layer in word_encoder_model.layers:
            layer.trainable = False

        for layer in word_decoder_model.layers:
            layer.trainable = False

        inputs = Input(shape=(max_sentence_length, 1))
        mask = Masking(pad_char_index)(inputs)

        word_embedding = TimeDistributed(word_encoder_model)(mask)

        encoder = LSTM(32, return_sequences=True)(word_embedding)
        encoder = LSTM(64, return_sequences=True)(encoder)
        encoder = LSTM(128, return_sequences=True)(encoder)

        encoded = LSTM(embedding_dim, return_sequences=False)(encoder)

        repeated = RepeatVector(max_sentence_length)(encoded)

        decoder = LSTM(128, return_sequences=True)(repeated)
        decoder = LSTM(256, return_sequences=True)(decoder)
        decoder = TimeDistributed(Dense(word2vec_embedding_dim))(decoder)

        outputs = TimeDistributed(word_decoder_model)(decoder)

        model = Model(inputs=inputs, outputs=outputs)
        self.model = model
        self.model.summary()
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate),
            metrics=["accuracy"],
        )

    def get_embedding_model(self):
        embedding_model_path = self.__get_rword2vec_model_save_path()
        embedding_model_json_path = embedding_model_path[:-2] + "json"
        with open(embedding_model_json_path) as json_file:
            model = model_from_json(json_file.read())
            model.load_weights(embedding_model_path)
            encoder_inputs = model.inputs[0]
            encoder_outputs = model.layers[3].get_output_at(0)

            encoder_model = Model(inputs=encoder_inputs,
                                  outputs=encoder_outputs)

            decoder_inputs = Input(
                shape=(encoder_outputs.get_shape().as_list()[1:]))
            decoder_outputs = Dense(model.outputs[0].get_shape().as_list()[
                                    1], activation="softmax")(decoder_inputs)

            decoder_model = Model(inputs=decoder_inputs,
                                  outputs=decoder_outputs)
            decoder_model.set_weights(model.layers[4].get_weights())

            return encoder_model, decoder_model

    def __get_rword2vec_model_save_path(self):
        model_dir = self.config["output"]["rword2vec"]["model_dir"]
        
        i = 1
        output_path = os.path.join(
            model_dir, self.language + "-rword2vec-%d.h5")
        while os.path.exists(output_path % i):
            i += 1
        else:
            i -= 1
        return output_path % i

    def __get_model_save_path(self):
        model_dir = self.config["output"]["sentence_rep"]["model_dir"]
        if not os.path.exists(model_dir):
            print("Creating folder: '%s'" % model_dir)
            os.makedirs(model_dir)
        i = 1
        output_path = os.path.join(
            model_dir, self.language + "-sentence_rep-%d.h5")
        while os.path.exists(output_path % i):
            i += 1
        return output_path % i

    def train(self, generator, validation_generator=None, save_model=True):

        epochs = self.config["hyper_params"]["sentence_representation"]["epochs"]
        steps = self.config["hyper_params"]["sentence_representation"]["steps"]
        val_steps = self.config["hyper_params"]["sentence_representation"]["val_steps"]
        self.model.fit_generator(generator, epochs=epochs, validation_data=validation_generator,
                                 verbose=1, steps_per_epoch=steps, validation_steps=val_steps)
        weights_path = self.__get_model_save_path()
        self.model.save_weights(weights_path)
        model_json = self.model.to_json()
        json_path = weights_path[:-2] + 'json'
        with open(json_path, "w+") as json_file:
            json_file.write(model_json)
