from tensorflow.keras.layers import Input, Reshape, Dense, Dot, Embedding
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam
import os

class Word2VecModel(object):
    def __init__(self, config, language):
        self.config = config
        self.language = language
        self.build()
    def build(self):
        embedding_dim = self.config["params"]["word2vec"]["embedding_dim"]
        vocab_size = self.config["params"]["vocab_size"]
        learning_rate = self.config["hyper_params"]["word2vec"]["lr"]
        
        target_input = Input(shape=(1,), name="target")
        context_input = Input(shape=(1,), name="context")
        embedding = Embedding(vocab_size, embedding_dim, input_length=1, name="embedding")
        
        target_embedding = embedding(target_input)
        context_embedding = embedding(context_input)

        target = Reshape((embedding_dim, 1))(target_embedding)
        context = Reshape((embedding_dim, 1))(context_embedding)

        similarity = Reshape((1,))(Dot(1, normalize=False)([target, context]))
        
        output = Dense(1, activation="sigmoid")(similarity)
        model = Model(inputs=[target_input, context_input], outputs=output)
        model.compile(loss="binary_crossentropy", optimizer=Adam(
            learning_rate), metrics=['accuracy'])
        self.model = model
    def train(self, generator, validation_generator=None, save_model=True):
        
        epochs = self.config["hyper_params"]["word2vec"]["epochs"]
        steps = self.config["hyper_params"]["word2vec"]["steps"]
        val_steps = self.config["hyper_params"]["word2vec"]["val_steps"]
        self.model.fit_generator(generator, epochs=epochs, validation_data=validation_generator, verbose=1, steps_per_epoch=steps, validation_steps=val_steps)
        weights_path = self.__get_model_save_path()
        self.model.save_weights(weights_path)
        model_json = self.model.to_json()
        json_path = weights_path[:-2] + 'json'
        with open(json_path, "w+") as json_file:
            json_file.write(model_json)
        

    def __get_model_save_path(self):
        model_dir = self.config["output"]["word2vec"]["model_dir"]
        if not os.path.exists(model_dir):
            print("Creating folder: '%s'"%model_dir)
            os.makedirs(model_dir)
        i = 1
        output_path = os.path.join(model_dir, self.language + "-word2vec-%d.h5")
        while os.path.exists(output_path % i):
            i += 1
        return output_path%i

class ReverseWord2VecModel(object):
    def __init__(self, config, language):
        self.config = config
        self.language = language
        self.build()
    def build(self):

        vocab_size = self.config["params"]["vocab_size"]
        learning_rate = self.config["hyper_params"]["word2vec"]["lr"]

        embedding_model = self.__get_embedding_model()

        inputs = embedding_model.inputs[0]
        embedding_outputs = embedding_model.layers[3].get_output_at(0)
        outputs = Reshape((embedding_outputs.get_shape().as_list()[1],), name="new_reshape")(
            embedding_outputs)
        outputs = Dense(vocab_size, activation="softmax")(outputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(
            learning_rate), metrics=['accuracy'])
        self.model = model
    def __get_embedding_model(self):
        word2vec_model_path = self.__get_word2vec_model_save_path()
        if os.path.exists(word2vec_model_path):
            json_path = word2vec_model_path[:-2] + "json"
            with open(json_path) as json_file:
                model = model_from_json(json_file.read())
                model.load_weights(word2vec_model_path)
                
                return model
                
        else:
            raise ValueError("Word2Vec model doesnot seem to be trained yet")
    
    def train(self, generator, validation_generator=None, save_model=True):
        
        epochs = self.config["hyper_params"]["rword2vec"]["epochs"]
        steps = self.config["hyper_params"]["rword2vec"]["steps"]
        val_steps = self.config["hyper_params"]["rword2vec"]["val_steps"]
        self.model.fit_generator(generator, epochs=epochs, validation_data=validation_generator, verbose=1, steps_per_epoch=steps, validation_steps=val_steps)
        weights_path = self.__get_model_save_path()
        self.model.save_weights(weights_path)
        model_json = self.model.to_json()
        json_path = weights_path[:-2] + 'json'
        with open(json_path, "w+") as json_file:
            json_file.write(model_json)
        

    def __get_word2vec_model_save_path(self):
        model_dir = self.config["output"]["word2vec"]["model_dir"]
        
        i = 1
        output_path = os.path.join(model_dir, self.language + "-word2vec-%d.h5")
        while os.path.exists(output_path % i):
            i += 1
        else:
            i-=1
        return output_path%i
        
    def __get_model_save_path(self):
        model_dir = self.config["output"]["rword2vec"]["model_dir"]
        if not os.path.exists(model_dir):
            print("Creating folder: '%s'" % model_dir)
            os.makedirs(model_dir)
        i = 1
        output_path = os.path.join(
            model_dir, self.language + "-rword2vec-%d.h5")
        while os.path.exists(output_path % i):
            i += 1
        return output_path % i
