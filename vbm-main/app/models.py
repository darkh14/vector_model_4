from db import db_processor
from typing import Optional
from sklearn.pipeline import Pipeline
from data_processing import data_validator, Reader, RowToColumn, Checker, NanProcessor, Shuffler
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
from datetime import datetime



logging.getLogger("vbm_data_processing_logger").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class Model:

    def __init__(self, model_id, parameters=None):
        self.model_id = model_id
        self.parameters = parameters.model_dump() if parameters else None
        self.data_filter = None
        self.ml_model = None
        self.scaler = None

    async def initialize(self):
        if not self.parameters:
            await self.read_model()

        if self.parameters:
            self.data_filter = self.parameters.get('data_filter')

    async def read_model(self):
        model_parameters = await db_processor.find_one('models', {'model_id': self.model_id})
        if model_parameters is not None:
            model_parameters['data_filter'] = pickle.loads(model_parameters['data_filter'])
            self.parameters = model_parameters
            model_binary = await db_processor.find_one('models_bin', {'model_id': self.model_id})
            if model_binary:
                
                scaler = Scaler(self.parameters)
                scaler.from_binary(model_binary['scaler'])
                self.scaler = scaler

                ml_model = NNModel(self.parameters)
                ml_model.from_binary(model_binary['model'])
                self.ml_model = ml_model

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

    async def predict(self, X, accounting_db=''):

        logger.info("Сhecking x data. Мodel id={}".format(self.model_id))
        data_validator.validate_raw_data(X)

        logger.info("Transforming and scaling x data. Мodel id={}".format(self.model_id))        
        X = pd.DataFrame(X)

        descr = self.get_aux_columns_descr(X)

        pipeline = Pipeline([
                            ('checker', Checker(self.parameters, for_predict=True)),
                            ('row_column_transformer', RowToColumn(self.parameters, for_predict=True)),
                            ('nan_processor', NanProcessor(self.parameters, for_predict=True)),
                            ('shuffler', Shuffler(self.parameters, for_predict=True)),
                            ('scaler', self.scaler),
                             ])

        X = pipeline.transform(X)

        logger.info("Predicting. Мodel id={}".format(self.model_id))

        y = self.ml_model.predict(X[self.parameters['x_columns']].to_numpy())
        X[self.parameters['y_columns']] = y

        logger.info("Reverse transforming result data. Мodel id={}".format(self.model_id))
        row_column_transformer = pipeline.named_steps['row_column_transformer']
        row_column_transformer.only_outers = True
        y = row_column_transformer.inverse_transform(X)
        
        y = await self.add_aux_columns_to_y_data(y, dims_descr=descr, accounting_db=accounting_db)

        y = y.to_dict(orient='records')

        logger.info("Predicting model is finished. Мodel id={}".format(self.model_id))
        return y
       
    def get_aux_columns_descr(self, data):

        dims = {}
        for dim in self.parameters['dimensions']:
            rows = data[['{}_id'.format(dim), '{}_kind'.format(dim), '{}_name'.format(dim)]].to_dict(orient='records')
            c_dict = {}
            for row in rows:
                if row['{}_id'.format(dim)] not in c_dict:
                    c_dict[row['{}_id'.format(dim)]] = {'kind': row['{}_kind'.format(dim)], 'name': row['{}_name'.format(dim)]}

            dims[dim] = c_dict    

        return dims

    async def add_aux_columns_to_y_data(self, data, dims_descr, accounting_db):

        all_columns = ['period']
        aux_columns = []
        indicators = []

        for dim in self.parameters['dimensions']:
            all_columns.append('{}_id'.format(dim))
            
            all_columns.append('{}_kind'.format(dim))
            all_columns.append('{}_name'.format(dim))     

            aux_columns.append('{}_kind'.format(dim))
            aux_columns.append('{}_name'.format(dim))                     

        all_columns.append('ind_id')
        all_columns.append('ind_kind')
        all_columns.append('ind_name')

        aux_columns.append('ind_kind')
        aux_columns.append('ind_name')   

        nums = []

        for column_settings in self.parameters['columns_descriptions'].values():
            if column_settings['outer']:
                for an in column_settings['analytics']:

                    all_columns.append['{}_id'.format(an)]
                    all_columns.append['{}_kinde'.format(an)]
                    all_columns.append['{}_name'.format(an)]

                    aux_columns.append['{}_kinde'.format(an)]
                    aux_columns.append['{}_name'.format(an)]     
                              
                if column_settings['num'] not in nums:
                    nums.append(column_settings['num'])

                if column_settings['indicator_id'] not in indicators:
                    indicators.append(column_settings['indicator_id'])

        for num in nums:
            all_columns.append('{}_value'.format(num))
            all_columns.append('{}_kind'.format(num))    

            aux_columns.append('{}_kind'.format(num))

        for col in aux_columns:
            data[col] = ''

        for dim, descr in dims_descr.items():
            data['{}_kind'.format(dim)] = data['{}_id'.format(dim)].apply(lambda x: descr[x]['kind'])
            data['{}_name'.format(dim)] = data['{}_id'.format(dim)].apply(lambda x: descr[x]['name'])     

        inds = {}
        ans = {}
        nums = {}
        for indicator_settings in self.parameters['indicators']:
            if not indicator_settings['outer']:
                continue
            
            data_row = await db_processor.find_one('raw_data', {'accounting_db': accounting_db, 'ind_id': indicator_settings['id']})
            c_ind = {'kind': data_row['ind_kind'], 'name': data_row['ind_name']}            

            inds[indicator_settings['id']] = c_ind

            for num in indicator_settings['numbers']:
                if num not in nums:
                    nums[num] = {'kind': data_row['{}_kind'.format(num)]}

            if indicator_settings['analytics']:
                data_filter = {'accounting_db': accounting_db, 'ind_id': indicator_settings['id']}
                for an in indicator_settings['analytics']:
                    data_filter['{}_id'.format(an)] = {'$in': list(data['{}_id'.format(an)].unique())}
                
                data_rows = await db_processor.find('raw_data', data_filter)
                
                for an in indicator_settings['analytics']:
                    c_an = ans.get(an) or {}
                    for row in data_rows:
                        if row['{}_id'.format(an)] not in c_an:
                            c_an[row['{}_id'.format(an)]] = {'name': row['{}_name'.format(an)], 'kind': row['{}_kind'.format(an)]}
                    ans[an] = c_an
                    
        data['ind_kind'] = data['ind_id'].apply(lambda x: inds[x]['kind'])
        data['ind_name'] = data['ind_id'].apply(lambda x: inds[x]['name'])   

        for num, num_setting in nums.items():
            data['{}_kind'.format(num)] = num_setting['kind'] 

        for an, descr in ans.items():
            data['{}_kind'.format(an)] = data['{}_id'.format(an)].apply(lambda x: descr.get(x, {}).get('kind', ''))
            data['{}_name'.format(an)] = data['{}_id'.format(an)].apply(lambda x: descr.get(x, {}).get('name', ''))  

        data = data[all_columns]

        return data


    async def write_to_db(self):

        parameters_to_db = self.parameters.copy()
        parameters_to_db['data_filter'] = pickle.dumps(parameters_to_db['data_filter'])
        await db_processor.insert_one('models', parameters_to_db, {'model_id': self.model_id})
        bin_data = {'model_id': self.model_id, 'scaler': self.scaler.get_binary(), 'model': self.ml_model.get_binary()}
        await db_processor.insert_one('models_bin', bin_data, {'model_id': self.model_id})

    async def get_info(self):

        if not self.parameters:
            return None
        
        result = {}
        result['model_id'] = self.model_id
        result['columns_descriptions'] = []
        for k, v in self.parameters['columns_descriptions'].items():
            descr = {'name': k, 
                     'analytics': v['analytics'],
                     'analytic_key': v['analytic_key'],
                     'period_shift': v['period_shift']}
            result['columns_descriptions'].append(descr)
        return result

    async def delete(self):
        await db_processor.delete_many('models', {'model_id': self.model_id})
        await db_processor.delete_many('models_bin', {'model_id': self.model_id})        

        self.parameters = None
        self.data_filter = None
        self.ml_model = None
        self.scaler = None


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
    def __init__(self, parameters=None):
        ...

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

    def __init__(self, parameters=None):
        self.nn = None

    def fit(self, X, y, parameters):
        self.nn = self.get_nn(X.shape[1], y.shape[1])

        self.nn.fit(X, y, epochs=parameters['epochs'])

    def predict(self, X):
        return self.nn.predict(X)

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
