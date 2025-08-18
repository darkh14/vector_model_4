from db import db_processor
from typing import Optional, Dict
from data_processing import data_procesor, data_validator, convert_dates_in_db_filter
from entities import ModelStatuses, ModelTypes, FIStatuses
from post_processing import post_processor
from errors import AlsoCalculatingException

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

    def __init__(self, model_id, model_type=None, parameters=None):
        self.model_type = model_type
        self.model_id = model_id
        self.parameters = parameters.model_dump() if parameters else None
        self.data_filter = None
        self.ml_model = None
        self.scaler = None
        self.status = ModelStatuses.CREATED
        self.error_text = ''
        self.initialized = False
        self.fi_status = FIStatuses.NOT_CALCULATED
        self.fi_error_text = ''
        self.feature_importances = {}  

    async def initialize(self, parameters=None):

        if self.initialized:
            return None
        
        if not self.model_type:
            await self.read_model()
        else:
            self.parameters = parameters if parameters else None            

        if self.parameters:
            self.data_filter = self.parameters.get('data_filter')

        self.initialized = True

    async def read_model(self):
        model_parameters = await db_processor.find_one('models', {'model_id': self.model_id})
        if model_parameters is not None:
            model_parameters['data_filter'] = pickle.loads(model_parameters['data_filter'])
            self.parameters = model_parameters
            status = model_parameters.get('status') or 'CREATED'
            fi_status = model_parameters.get('fi_status') or 'NOT_CALCULATED'
            model_type = model_parameters.get('model_type')
            self.status = ModelStatuses(status)
            self.fi_status = FIStatuses(fi_status)
            self.model_type = ModelTypes(model_type) if model_type else None
            self.error_text = model_parameters['error_text']
            self.fi_error_text = model_parameters['fi_error_text'] 

            self.feature_importances = model_parameters['feature_importances']         

            model_bin = model_parameters['model']
            scaler_bin = model_parameters['scaler']

            if model_bin:
                ml_model = self._get_ml_model(self.model_type, self.parameters)
                ml_model.from_binary(model_bin)
                self.ml_model = ml_model
            
            if scaler_bin:
                scaler = Scaler(self.parameters)
                scaler.from_binary(scaler_bin)
                self.scaler = scaler

    async def fit(self, data_filter: Optional[dict] = None):
        
        try:
            if self.status == ModelStatuses.FITTING:
                raise AlsoCalculatingException('Model is fitting now!')
            
            if not self.status in [ModelStatuses.CREATED, ModelStatuses.ERROR, ModelStatuses.READY]:
                raise ValueError('Model is not ready to be fit!')

            if self.status == ModelStatuses.ERROR:
                self.initialized = False
                self.initialize(self.parameters)              

            self.status = ModelStatuses.FITTING
            self.error_text = ''
            self.fi_error_text = ''
            self.fi_status = FIStatuses.NOT_CALCULATED

            self.feature_importances = {}         

            logger.info("Reading data from db. Мodel id={}".format(self.model_id))
            self.scaler = Scaler(self.parameters)

            X_y = await data_procesor.get_dataset(self.model_id, self.scaler, self.parameters, data_filter)
            X, y = X_y[self.parameters['x_columns']].to_numpy(), X_y[self.parameters['y_columns']].to_numpy()

            logger.info("Fitting model. Мodel id={}".format(self.model_id))
            self.ml_model = self._get_ml_model(self.model_type, self.parameters)
            self.ml_model.fit(X, y, parameters=self.parameters)

            logger.info("Saving model and fitting parameters to db. Мodel id={}".format(self.model_id))
            await self.write_to_db()
            self.status = ModelStatuses.READY
            logger.info("Fitting model is finished. Мodel id={}".format(self.model_id))
        
        except AlsoCalculatingException as e:
            raise e
        except Exception as e:
            self.status = ModelStatuses.ERROR
            self.fi_status= FIStatuses.NOT_CALCULATED
            self.error_text = str(e)

            await self.write_to_db()
            raise e            

        return True

    async def predict(self, X):

        logger.info("Сhecking x data. Мodel id={}".format(self.model_id))
        data_validator.validate_raw_data(X)

        logger.info("Transforming and scaling x data. Мodel id={}".format(self.model_id))        
        X = pd.DataFrame(X)

        descr = self.get_aux_columns_descr(X)

        X = await data_procesor.transform_dataset(X, self.model_id, self.scaler, self.parameters)

        logger.info("Predicting. Мodel id={}".format(self.model_id))

        y = self.ml_model.predict(X[self.parameters['x_columns']].to_numpy())
        X[self.parameters['y_columns']] = y

        logger.info("Reverse transforming result data. Мodel id={}".format(self.model_id))
        pipeline = data_procesor.fit_pipelines[self.model_id]
        row_column_transformer = pipeline.named_steps['row_column_transformer']
        row_column_transformer.only_outers = True
        y = row_column_transformer.inverse_transform(X)
        
        y = await self.add_aux_columns_to_y_data(y, dims_descr=descr)

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

    async def add_aux_columns_to_y_data(self, data, dims_descr):

        data_filter = self.parameters.get('data_filter')
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

        all_columns.append('accounting_db')  

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
            
            if not data_filter:
                c_data_filter = {}
            else:
                c_data_filter = convert_dates_in_db_filter(data_filter.copy())

            c_data_filter['ind_id'] = indicator_settings['id']

            data_row = await db_processor.find_one('raw_data', c_data_filter)
            if data_row:
                c_ind = {'kind': data_row['ind_kind'], 'name': data_row['ind_name']}            

                inds[indicator_settings['id']] = c_ind

                for num in indicator_settings['numbers']:
                    if num not in nums:
                        nums[num] = {'kind': data_row['{}_kind'.format(num)]}

            if indicator_settings['analytics']:

                if not data_filter:
                    c_data_filter = {}
                else:
                    c_data_filter = convert_dates_in_db_filter(data_filter.copy()) 

                c_data_filter['ind_id'] = indicator_settings['id']
                for an in indicator_settings['analytics']:
                    c_data_filter['{}_id'.format(an)] = {'$in': list(data['{}_id'.format(an)].unique())}
                
                data_rows = await db_processor.find('raw_data', c_data_filter)

                if data_rows:
                    for an in indicator_settings['analytics']:
                        c_an = ans.get(an) or {}
                        for row in data_rows:
                            if row['{}_id'.format(an)] not in c_an:
                                c_an[row['{}_id'.format(an)]] = {'name': row['{}_name'.format(an)], 'kind': row['{}_kind'.format(an)]}
                        ans[an] = c_an
                    
        data['ind_kind'] = data['ind_id'].apply(lambda x: inds[x].get('kind', ''))
        data['ind_name'] = data['ind_id'].apply(lambda x: inds[x].get('name', ''))   

        for num, num_setting in nums.items():
            data['{}_kind'.format(num)] = num_setting['kind'] 

        for an, descr in ans.items():
            data['{}_kind'.format(an)] = data['{}_id'.format(an)].apply(lambda x: descr.get(x, {}).get('kind', ''))
            data['{}_name'.format(an)] = data['{}_id'.format(an)].apply(lambda x: descr.get(x, {}).get('name', ''))  

        data_row = await db_processor.find_one('raw_data', data_filter)
        if data_row:
            data['accounting_db'] = data_row['accounting_db']
        else:
            data['accounting_db'] = ''           

        data = data[all_columns]

        return data

    async def write_to_db(self):

        parameters_to_db = self.parameters.copy() if self.parameters else {}
        parameters_to_db['data_filter'] = pickle.dumps(parameters_to_db['data_filter']) if parameters_to_db['data_filter'] else b''
        parameters_to_db['status'] = self.status.value
        parameters_to_db['fi_status'] = self.fi_status.value        
        parameters_to_db['model_type'] = self.model_type.value if self.model_type else ''

        model_bin = self.ml_model.get_binary() if self.ml_model else None
        scaler_bin = self.scaler.get_binary() if self.scaler else None
        parameters_to_db['model'] = model_bin
        parameters_to_db['scaler'] = scaler_bin

        parameters_to_db['error_text'] = self.error_text
        parameters_to_db['fi_error_text'] = self.fi_error_text

        parameters_to_db['feature_importances'] = self.feature_importances       

        await db_processor.insert_one('models', parameters_to_db, {'model_id': self.model_id})

    async def get_info(self):

        if not self.parameters:
            return None
        
        result = {}
        result['model_id'] = self.model_id
        result['columns_descriptions'] = []

        if 'columns_descriptions' in self.parameters:
            for k, v in self.parameters['columns_descriptions'].items():
                descr = {'name': k, 
                        'analytics': v['analytics'],
                        'analytic_key': v['analytic_key'],
                        'period_shift': v['period_shift']}
                result['columns_descriptions'].append(descr)

        result['status'] = self.status.value
        result['fi_status'] = self.fi_status.value
        result['error_text'] = self.error_text
        result['fi_error_text'] = self.fi_error_text        
        return result

    async def delete(self):
        await db_processor.delete_many('models', {'model_id': self.model_id})     

        self.parameters = None
        self.data_filter = None
        self.ml_model = None
        self.scaler = None
        self.status = ModelStatuses.CREATED
        self.fi_status = FIStatuses.NOT_CALCULATED
        self.feature_importances = {}
        self.error_text = ''
        self.fi_error_text = ''
        self.initialized = False

    def _get_ml_model(self, model_type, parameters):
        ml_model_classes = [el for el in MlModel.__subclasses__() if el.model_type == model_type]

        if not ml_model_classes:
            raise ValueError('Model type "{}" is not supported!'.format(model_type))
        
        return ml_model_classes[0](parameters)


class ModelManager:

    def __init__(self):
        self.models = []

    def get_model(self, model_id, model_type=None):
        c_models = [el for el in self.models if el['model_id'] == model_id]
        if c_models:
            model  = c_models[0]['model']
        elif model_type:
            model = Model(model_id, model_type=model_type) 
            self.models.append({'model_id': model_id, 'model_type': model_type, 'model': model})     
        else:
            model = None  
        
        return model

    async def read_models(self):

        self.models = []
        models_from_db = await db_processor.find('models')
        for model_from_db in models_from_db:
            model = Model(model_from_db['model_id'])
            await model.initialize()

            self.models.append({'model_id': model_from_db['model_id'],
                                'model_type': model_from_db['model_type'],
                                'model': model})

    async def _get_model_from_db(self, model_id):
        model = Model(model_id)
        await model.initialize()

        if not model.parameters:
            model = None

        return model

    async def delete_model(self, model_id):
        model = self.get_model(model_id)
        if not model:
            raise ValueError('Model id "{}" not found'.format(model_id))

        self.models = [el for el in self.models if el['model_id'] != model_id]
        await model.delete()


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
    model_type = None
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
    model_type = ModelTypes.nn
    def __init__(self, parameters=None):
        self.nn = None

    def fit(self, X, y, parameters):
        self.nn = self.get_nn(X.shape[1], y.shape[1])
        epochs = parameters.get('epochs') or 300
        self.nn.fit(X, y, epochs=epochs)

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


class PfModel(MlModel):
    model_type = ModelTypes.pf
    def __init__(self, parameters=None):
        self.power = parameters.get('power') or 2
        self.pf = None
        self.linear = None

    def fit(self, X, y, parameters):
        self.linear = self.get_linear(X.shape[1], y.shape[1])
        self.pf = self.get_pf()

        X_pf = self.pf.fit_transform(X)

        return self.linear.fit(X_pf, y)

    def predict(self, X):
        X_pf = self.pf.transform(X)        
        return self.linear.predict(X_pf)

    def get_pf(self):
        return PolynomialFeatures(degree=self.power, interaction_only=True, include_bias=True)
    
    def get_linear(self, input_number, output_number):
        return LinearRegression()

    def get_binary(self):

        model_data = {'linear': self.linear, 'pf': self.pf}
        model_bin = pickle.dumps(model_data)

        return model_bin
    
    def from_binary(self, model_bin):

        model_data = pickle.loads(model_bin)
        self.pf = model_data['pf']
        self.linear = model_data['linear']


model_manager = ModelManager()