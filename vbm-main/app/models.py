from db import db_processor
from typing import Optional, Dict
from data_processing import data_procesor, data_validator, convert_dates_in_db_filter
from entities import ModelStatuses, ModelTypes, FIStatuses
from post_processing import post_processor
from errors import AlsoCalculatingException
import zipfile
import json
import shutil

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import asyncio
import pandas as pd
import numpy as np
import pickle
from abc import ABC, abstractmethod
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam

from catboost import CatBoostRegressor
from calculation import DirectModel

import tempfile
import logging
from datetime import datetime, timezone
from uuid import uuid4
import os
from config import TEMP_FOLDER


logging.getLogger("vbm_data_processing_logger").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class Model:

    def __init__(self, model_id, model_type=None, parameters=None):
        self.model_type = model_type
        self.model_id = model_id
        self.parameters = parameters.model_dump() if parameters else None
        self.data_filter = None
        self.ml_model = None
        self.use_period_number = False
        self.scaler = None
        self.status = ModelStatuses.CREATED
        self.error_text = ''
        self.initialized = False
        self.fi_status = FIStatuses.NOT_CALCULATED
        self.fi_error_text = ''
        self.feature_importances = {}
        self.metrics = {}
        self.fitting_start_date = None
        self.fitting_date = None
        self.extra_x_predict_columns = []
        self.model_path = '../models/{}.mdl'.format(self.model_id)
        self.use_scaler = True

    async def initialize(self, parameters=None):

        if self.initialized:
            return None
        logger.info('initialize model {}, {}'.format(self.model_id, self.model_type))
        if not self.model_type:
            await self.read_model()
        else:
            self.parameters = parameters if parameters else None            

        if self.parameters:
            self.data_filter = self.parameters.get('data_filter')
            self.use_period_number = self.parameters.get('use_period_number', False)
            self.extra_x_predict_columns = self.parameters.get('extra_x_predict_columns', []) 
            self.use_scaler = self.parameters.get('use_scaler', False)           

        self.initialized = True

    async def read_model(self):
        model_parameters = await db_processor.find_one('models', {'model_id': self.model_id})
        if model_parameters is not None:
            model_parameters['data_filter'] = pickle.loads(model_parameters['data_filter'])
            self.parameters = model_parameters
            status = model_parameters.get('status') or 'CREATED'
            fi_status = model_parameters.get('fi_status') or 'NOT_CALCULATED'
            model_type = model_parameters.get('model_type')
            self.use_period_number = model_parameters.get('use_period_number', False)             
            self.status = ModelStatuses(status)
            self.fi_status = FIStatuses(fi_status)
            self.model_type = ModelTypes(model_type) if model_type else None
            self.error_text = model_parameters['error_text']
            self.fi_error_text = model_parameters['fi_error_text'] 

            self.extra_x_predict_columns = model_parameters.get('extra_x_predict_columns') or []
            self.use_scaler = model_parameters.get('use_scaler', False)

            self.feature_importances = model_parameters['feature_importances'] 

            self.metrics = model_parameters.get('metrics', {})    
            
            if not os.path.isdir('../models'):
                os.mkdir('../models')
            logger.info('-----Model path = {}, {}'.format(self.model_path, os.path.isfile(self.model_path)))                
            model_bin = None
            if os.path.isfile(self.model_path):
                logger.info('-----Model path = {}'.format(self.model_path))
                with open(self.model_path, 'rb') as fp:
                    model_bin = pickle.load(fp)


            scaler_bin = model_parameters['scaler']

            self.fitting_start_date = model_parameters.get('fitting_start_date')
            self.fitting_date = model_parameters.get('fitting_date')         

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
            self.fitting_start_date = datetime.now(tz=timezone.utc)

            self.feature_importances = {}      

            logger.info("Reading data from db. Мodel id={}".format(self.model_id))
            self.scaler = Scaler(self.parameters)

            X_y = await data_procesor.get_dataset(self.model_id, self.scaler if self.use_scaler else None, self.parameters, data_filter)
            X, y = X_y[self.parameters['x_columns']].to_numpy(), X_y[self.parameters['y_columns']].to_numpy()

            logger.info("Fitting model. Мodel id={}".format(self.model_id))
            self.ml_model = self._get_ml_model(self.model_type, self.parameters)
            self.ml_model.fit(X, y, parameters=self.parameters)
            self.status = ModelStatuses.READY
            self.fitting_date = datetime.now(tz=timezone.utc)            

            logger.info("Testing model and getting metrics. Мodel id={}".format(self.model_id))

            self.metrics = self._get_metrics(X_y)

            if self.use_scaler and self.scaler:
                self.parameters['scaler'] = self.scaler.get_binary()

            self.parameters['model'] = self.ml_model.get_binary()

            self.parameters['fitting_date'] = self.fitting_date
            self.parameters['fitting_start_date'] = self.fitting_start_date     
                    
            logger.info("Saving model and fitting parameters to db. Мodel id={}".format(self.model_id))

            await self.write_to_db()            
            logger.info("Fitting model is finished. Мodel id={}".format(self.model_id))
        
        except AlsoCalculatingException as e:
            raise e
        except Exception as e:
            self.status = ModelStatuses.ERROR
            self.fi_status= FIStatuses.NOT_CALCULATED
            self.error_text = str(e)
            self.fitting_date = None             

            await self.write_to_db()
            raise e            

        return True

    async def predict(self, X):

        logger.info("Сhecking x data. Мodel id={}".format(self.model_id))
        data_validator.validate_raw_data(X)

        logger.info("Transforming and scaling x data. Мodel id={}".format(self.model_id))        
        X = pd.DataFrame(X)

        descr = self.get_aux_columns_descr(X)

        X = await data_procesor.transform_dataset(X, self.model_id, self.scaler if self.use_scaler else None, self.parameters)

        logger.info("Predicting. Мodel id={}".format(self.model_id))

        x_columns = self.parameters['x_columns'].copy()
        if self.extra_x_predict_columns:
            for col, from_col in self.extra_x_predict_columns:
                X[col] = X[from_col]
                x_columns.append(col)

        if (hasattr(self.ml_model, 'cb')
                and hasattr(self.ml_model.cb, 'model_bid') 
                and len(self.extra_x_predict_columns) == 1):
            self.ml_model.cb.model_bid = True
        if self.extra_x_predict_columns:
            dims = list(X['dim_0_id'].unique())

            X_y_list = []
            for dim in dims:
                X_y_dim = X.loc[X['dim_0_id'] == dim].copy()
                
                y_dim = self.ml_model.predict(X_y_dim[x_columns].to_numpy())
                X_y_dim[self.parameters['y_columns'][0]] = y_dim
                X_y_list.append(X_y_dim)
            X_y = pd.concat(X_y_list, axis=0)
            X_y = X_y.sort_index()

            y = X_y[self.parameters['y_columns']]
        else:
            y = self.ml_model.predict(X[x_columns].to_numpy())
            y = pd.DataFrame(y, columns=self.parameters['y_columns'])

        for col in self.parameters['y_columns']:
            X[col] = y[col]
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

                    all_columns.append('{}_id'.format(an))
                    all_columns.append('{}_kind'.format(an))
                    all_columns.append('{}_name'.format(an))

                    aux_columns.append('{}_kind'.format(an))
                    aux_columns.append('{}_name'.format(an))   
                              
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

        parameters_to_db['use_period_number'] = self.use_period_number         

        model_bin = self.ml_model.get_binary() if self.ml_model else None
        scaler_bin = self.scaler.get_binary() if self.scaler else None
        # parameters_to_db['model'] = model_bin
        parameters_to_db['scaler'] = scaler_bin

        parameters_to_db['error_text'] = self.error_text
        parameters_to_db['fi_error_text'] = self.fi_error_text

        parameters_to_db['feature_importances'] = self.feature_importances
        parameters_to_db['metrics'] = self.metrics  

        parameters_to_db['fitting_start_date'] = self.fitting_start_date
        parameters_to_db['fitting_date'] = self.fitting_date
        parameters_to_db['use_scaler'] = self.use_scaler

        parameters_to_db['extra_x_predict_columns'] = self.extra_x_predict_columns or None

        await db_processor.insert_one('models', parameters_to_db, {'model_id': self.model_id})

        if not os.path.isdir('../models'):
            os.mkdir('../models')

        with open(self.model_path, 'wb') as fp:
            pickle.dump(model_bin, fp)               

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
        result['metrics'] = self.metrics

        result['fitting_start_date'] = self.fitting_start_date
        result['fitting_date'] = self.fitting_date

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
        
        if os.path.isfile(self.model_path):
            os.remove(self.model_path)

    def _get_ml_model(self, model_type, parameters):
        ml_model_classes = [el for el in MlModel.__subclasses__() if el.model_type == model_type]

        if not ml_model_classes:
            raise ValueError('Model type "{}" is not supported!'.format(model_type))
        
        return ml_model_classes[0](parameters)

    def _get_metrics(self, X_y):
        y_true = X_y[self.parameters['y_columns']].to_numpy()
        y_pred = self.ml_model.predict(X_y[self.parameters['x_columns']])

        rmse = self._calculate_rmse(y_true, y_pred)
        mspe = self._calculate_mspe(y_true, y_pred)

        return {'RMSE': rmse, 'MSPE': mspe}
    
    async def save(self):
        if self.status != ModelStatuses.READY:
            raise ValueError('Model is not ready to be saved. Fit model before!')
        
        path_scaler = 'scaler_{}.mdl'.format(self.model_id)
        path_model = 'engine_{}.mdl'.format(self.model_id)
        path_parameters = 'parameters_{}.json'.format(self.model_id)

        path_zip = 'model_{}.zip'.format(self.model_id)

        model_parameters = self.parameters.copy()
        if self.scaler:
            with open(path_scaler, 'wb') as fp:
                fp.write(self.scaler.get_binary())
            model_parameters['is_scaler'] = True
        else:
            model_parameters['is_scaler'] = False

        if self.ml_model:
            with open(path_model, 'wb') as fp:
                fp.write(self.ml_model.get_binary())  
            model_parameters['is_model'] = True
        else:
            model_parameters['is_model'] = False

        if 'model' in model_parameters:
            del model_parameters['model']
        if 'scaler' in model_parameters:
            del model_parameters['scaler']

        model_parameters['extra_x_predict_columns'] = self.extra_x_predict_columns or None        

        model_parameters['fitting_date'] = (model_parameters['fitting_date'].strftime('%d.%m.%Y %H:%M:%S') 
                            if model_parameters['fitting_date'] else None)
        model_parameters['fitting_start_date'] = (model_parameters['fitting_start_date'].strftime('%d.%m.%Y %H:%M:%S') 
                            if model_parameters['fitting_start_date'] else None) 
        model_parameters['model_type'] = self.model_type.value    
        model_parameters['status'] = self.status.value
        model_parameters['fi_status'] = self.fi_status.value 

        model_parameters['use_scaler'] = self.use_scaler 
        model_parameters['feature_importances'] = self.feature_importances       

        with open(path_parameters, 'w', encoding='utf-8') as fp:
            json.dump(model_parameters, fp)                       

        zip_object = zipfile.ZipFile(path_zip, 'w')

        zip_object.write(path_scaler, arcname='scaler.mdl', compress_type=zipfile.ZIP_DEFLATED)
        zip_object.write(path_model, arcname='model.mdl', compress_type=zipfile.ZIP_DEFLATED)
        zip_object.write(path_parameters, arcname='parameters.json', compress_type=zipfile.ZIP_DEFLATED)        

        zip_object.close()

        os.remove(path_scaler)
        os.remove(path_model)
        os.remove(path_parameters)        

        return path_zip

    async def load(self, model_data, model_path):

        if os.path.splitext(model_path)[1] != '.zip':
            raise ValueError('Model file must be zip archive')

        path_zip = 'model_{}.zip'.format(self.model_id)

        with open(path_zip, 'wb+') as fp:
            content = model_data.read()
            fp.write(content)

        zip_object = zipfile.ZipFile(path_zip, 'r')
        zipped_files = [el.filename for el in zip_object.filelist]

        if 'parameters.json' not in zipped_files:
            raise ValueError('Model file must contain file parameters.json')

        path_folder = 'model_{}_temp'.format(self.model_id)
        path_scaler = os.path.join(path_folder, 'scaler.mdl')
        path_model = os.path.join(path_folder, 'model.mdl')
        path_paramerters = os.path.join(path_folder, 'parameters.json')

        is_model = False
        if 'model.mdl' in zipped_files:
            is_model = True
            zip_object.extract('model.mdl', path=path_folder)

        is_scaler = False
        if 'scaler.mdl' in zipped_files:
            is_scaler = True
            zip_object.extract('scaler.mdl', path=path_folder)        

        zip_object.extract('parameters.json', path=path_folder)

        zip_object.close()

        with open(path_paramerters, 'r', encoding='utf-8') as fp:
            model_parameters = json.load(fp)
        
        model_parameters['fitting_date'] = (datetime.strptime(model_parameters['fitting_date'], '%d.%m.%Y %H:%M:%S') 
                            if model_parameters['fitting_date'] else None)
        model_parameters['fitting_start_date'] = (datetime.strptime(model_parameters['fitting_start_date'], '%d.%m.%Y %H:%M:%S') 
                            if model_parameters['fitting_start_date'] else None)
        
        self.parameters = model_parameters

        status = model_parameters.get('status') or 'CREATED'
        fi_status = model_parameters.get('fi_status') or 'NOT_CALCULATED'
        model_type = model_parameters.get('model_type')
        self.status = ModelStatuses(status)
        self.fi_status = FIStatuses(fi_status)
        self.model_type = ModelTypes(model_type) if model_type else None
        self.use_period_number = model_parameters.get('use_period_number', False)         
        self.error_text = ''
        self.fi_error_text = ''

        self.fitting_start_date = model_parameters.get('fitting_start_date')
        self.fitting_date = model_parameters.get('fitting_date')   

        self.use_scaler = model_parameters.get('use_scaler', False)   
        self.extra_x_predict_columns = model_parameters.get('extra_x_predict_columns', [])

        self.feature_importances = model_parameters.get('feature_importances') 

        self.metrics = model_parameters.get('metrics')     

        if is_scaler:
            with open(path_scaler, 'rb') as fp:
                scaler_bin = fp.read()
            if not self.scaler:
                self.scaler = Scaler(self.parameters)
            self.scaler.from_binary(scaler_bin)

            self.parameters['scaler'] = self.scaler
        
        if is_model:
            with open(path_model, 'rb') as fp:
                model_bin = fp.read()

            ml_model = self._get_ml_model(self.model_type, self.parameters)
            ml_model.from_binary(model_bin)
            self.ml_model = ml_model            
        
        await self.write_to_db()

        os.remove(path_zip)
        shutil.rmtree(path_folder)        

    @staticmethod
    def _calculate_mspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates mean squared percentage error metric
        :param y_true: real output data
        :param y_pred: predicted output data
        :return: value of calculated metric
        """
        eps = np.zeros(y_true.shape)
        eps[:] = 0.0001
        y_p = np.c_[abs(y_true), abs(y_pred), eps]
        y_p = np.max(y_p, axis=1).reshape(-1, 1)

        return np.sqrt(np.nanmean(np.square(((y_true - y_pred) / y_p))))

    @staticmethod
    def _calculate_rmse(y_true, y_pred) -> float:
        """
        Calculates root mean squared error metric
        :param y_true: real output data
        :param y_pred: predicted output data
        :return: value of calculated metric
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))    


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
            return None  
        
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
        self.columns_to_scale = []

    def get_x_columns(self):
        result = [el for el in self.parameters['x_columns'] if el != 'period_number']
        if self.parameters.get('extra_x_predict_columns'):
            result = result + [el[0] for el in self.parameters['extra_x_predict_columns']]

        return result
        
    def fit(self, x: pd.DataFrame,
            y: Optional[pd.DataFrame] = None):
        """
        Saves engine parameters to scale data
        :param x: data to scale
        :param y: None
        :return: self scaling object
        """

        self.columns_to_scale = self.get_x_columns()
        data = x[self.columns_to_scale]
        self._scaler_engine = MinMaxScaler()
        self._scaler_engine.fit(data)

        return self        
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms data after saving scaler parameters
        :param x: data before scaling
        :return: data after scaling
        """
        self.columns_to_scale = self.get_x_columns()
        result = x.copy()
        prev_x_columns = list(result.columns)
        for col in self.columns_to_scale:
            if col not in prev_x_columns:
                result[col] = 0
            
        result[self.columns_to_scale] = self._scaler_engine.transform(result[self.columns_to_scale])

        return result

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transforms data after predicting to get real (unscaled) result
        :param x: data before unscaling
        :return: data after unscaling
        """
        self.columns_to_scale = self.get_x_columns()
        result = x.copy()

        result[self.columns_to_scale] = self._scaler_engine.inverse_transform(result[self.columns_to_scale])

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

        # with tempfile.TemporaryFile(suffix='.keras') as fp:
        #     self.nn.save(fp.name)
        #     fp.close()

        #     with open(fp.name, mode='rb') as f:
        #         model_bin = f.read()
        file_id = uuid4()
        file_path = os.path.join(TEMP_FOLDER, '{}.keras'.format(file_id))
        self.nn.save(file_path)

        with open(file_path, mode='rb') as f:
            model_bin = f.read()   

        os.remove(file_path)     

        return model_bin
    
    def from_binary(self, model_bin):

        # with tempfile.TemporaryFile(suffix='.keras') as fp:
        #     with open(fp.name, mode='wb') as f:
        #         f.write(model_bin)   
                     
        #     self.nn = load_model(fp.name)
        #     fp.close() 

        file_id = uuid4()
        file_path = os.path.join(TEMP_FOLDER, '{}.keras'.format(file_id))
        
        with open(file_path, mode='wb') as f:
            f.write(model_bin)

        self.nn = load_model(file_path)

        os.remove(file_path)


class CBModel(MlModel):
    model_type = ModelTypes.cb
    def __init__(self, parameters=None):
        self.cb: CatBoostRegressor = None

    def fit(self, X, y, parameters):
        iterations = parameters.get('iterations') or 300
        self.cb = self.get_cb(iterations)        
        self.cb.fit(X, y)

    def predict(self, X):
        return self.cb.predict(X)

    def get_cb(self, iterations):
        depth = 16
        learning_rate=1

        cb = CatBoostRegressor(iterations, depth=depth, learning_rate=learning_rate)
        return cb

    def get_binary(self):

        # file_id = uuid4()
        # file_path = os.path.join(TEMP_FOLDER, '{}.cb'.format(file_id))
        # self.cb.save_model(file_path)

        # with open(file_path, mode='rb') as f:
        #     model_bin = f.read()   

        # os.remove(file_path)     
        model_bin = pickle.dumps(self.cb)

        return model_bin
    
    def from_binary(self, model_bin):

        # file_id = uuid4()
        # file_path = os.path.join(TEMP_FOLDER, '{}.keras'.format(file_id))
        
        # with open(file_path, mode='wb') as f:
        #     f.write(model_bin)

        # self.cb = CatBoostRegressor.load_model(file_path)

        # os.remove(file_path)

        self.cb = pickle.loads(model_bin)


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