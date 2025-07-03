from db import db_processor
from typing import Optional
from sklearn.pipeline import Pipeline
from data_processing import Reader, RowToColumn, Checker, NanProcessor, Shuffler
from sklearn.preprocessing import MinMaxScaler
import asyncio
import pandas as pd
import pickle
from abc import ABC, abstractmethod
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import tempfile
import logging

logging.getLogger("vbm_data_processing_logger").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class Model:

    def __init__(self, model_id, parameters=None):
        self.model_id = model_id

        self.parameters = parameters.model_dump() if parameters else None

    async def initialize(self):
        if not self.parameters:
            self.parameters = await self.read_model(self.model_id)
        self.data_filter = self.parameters['data_filter']


    @staticmethod
    async def read_model(model_id):
        return db_processor.find_one('models', {'model_id': model_id})
    
    @staticmethod
    async def write_model(model_id, model_data):
        return db_processor.insert_one('models', model_data, {'model_id': model_id})
    
    async def fit(self, data_filter: Optional[dict] = None):
        
        logger.info("Reading data from db. Мodel id={}".format(self.model_id))
        X_y = await Reader().read(data_filter)
        logger.info("Transforming and checking data. Мodel id={}".format(self.model_id))
        pipeline = Pipeline([
                            ('checker', Checker(self.parameters)),
                             ('row_column_transformer', RowToColumn(self.parameters)),
                             ('nan_processor', NanProcessor(self.parameters)),
                             ('shuffler', Shuffler(self.parameters)),
                             ('scaler', Scaler(self.parameters)),
                             ])

        self.scaler = pipeline.named_steps['scaler']

        X_y = pipeline.fit_transform(X_y, [])

        X, y = X_y[self.parameters['x_columns']].to_numpy(), X_y[self.parameters['y_columns']].to_numpy()

        logger.info("Fitting model. Мodel id={}".format(self.model_id))
        self.ml_model = NNModel()
        f_parameters = {'epochs': 300}
        self.ml_model.fit(X, y, f_parameters)

        logger.info("Saving model and fitting parameters to db. Мodel id={}".format(self.model_id))
        await self.write_to_db()
        logger.info("Fitting model is finished. Мodel id={}".format(self.model_id))

    async def write_to_db(self):

        parameters_to_db = self.parameters.copy()
        del parameters_to_db['data_filter']
        await db_processor.insert_one('models', parameters_to_db, {'model_id': self.model_id})
        bin_data = {'model_id': self.model_id, 'scaler': self.scaler.get_binary(), 'model': self.ml_model.get_binary()}
        await db_processor.insert_one('models_bin', bin_data, {'model_id': self.model_id})


class Scaler:

    def __init__(self, parameters):
        self.parameters = parameters
        self._scaler_engine = None

    def fit(self, x: pd.DataFrame,
            y: Optional[pd.DataFrame] = None):
        """
        Saves engine parameters to scale data
        :param x: data to scale
        :param y: None
        :return: self scaling object
        """

        data = x[self.parameters['x_columns']]
        self._scaler_engine = MinMaxScaler()
        self._scaler_engine.fit(data)

        return self        
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms data after saving scaler parameters
        :param x: data before scaling
        :return: data after scaling
        """

        result = x.copy()
        result[self.parameters['x_columns']] = self._scaler_engine.transform(result[self.parameters['x_columns']])

        return result

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transforms data after predicting to get real (unscaled) result
        :param x: data before unscaling
        :return: data after unscaling
        """

        result = x.copy()

        result[self.parameters['x_columns']] = self._scaler_engine.inverse_transform(result[self.parameters['x_columns']])

        return result

    def get_binary(self):
        return pickle.dumps(self._scaler_engine)
    
    def from_binary(self, scaler_bin):
        self._scaler_engine = pickle.loads(scaler_bin)


class MlModel(ABC):

    @abstractmethod
    def fit(self, X, y, parameters=None):
        ...

    @abstractmethod
    def predict(self, X, parameters=None):
        ...
        
    @abstractmethod
    def get_binary(self):
        ...

    @abstractmethod
    def from_binary(self, model_bin):
        ...


class NNModel(MlModel):

    def __init__(self):
        self.nn = None

    def fit(self, X, y, parameters):
        self.nn = self.get_nn(X.shape[1], y.shape[1])

        self.nn.fit(X, y, epochs=parameters['epochs'])

    def predict(self, X):
        pass

    def get_nn(self, input_number, output_number):
        nn = Sequential()
        
        nn.add(Input((input_number,), name='input'))
        nn.add(Dense(300, activation="relu", name='dense_1'))
        nn.add(Dense(250, activation="relu", name='dense_2'))
        nn.add(Dense(100, activation="relu",  name='dense_3'))
        nn.add(Dense(30, activation="relu", name='dense_4'))
        nn.add(Dense(output_number, activation="linear", name='dense_last'))

        nn.compile(optimizer=Adam(learning_rate=0.001), loss='MeanSquaredError',
                      metrics=['RootMeanSquaredError'])

        return nn

    def get_binary(self):

        with tempfile.TemporaryFile(delete=False, suffix='.keras') as fp:
            self.nn.save(fp.name)
            fp.close()

            with open(fp.name, mode='rb') as f:
                model_bin = f.read()

        return model_bin
    
    def from_binary(self, model_bin):

        with tempfile.TemporaryFile(delete=False, suffix='.keras') as fp:
            with open(fp.name, mode='wb') as f:
                f.write(model_bin)   
                     
            self.nn = load_model(fp.name)
            fp.close()
