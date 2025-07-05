from storage import task_storage
import logging
import asyncio
import zipfile
import os
import pandas as pd
from config import TEMP_FOLDER
import json
from db import db_processor
from entities import RawqDataStr
from pydantic import TypeAdapter, ValidationError
from pydantic_core import InitErrorDetails
from typing import List
from collections import defaultdict
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
import asyncio
import uuid 

from pydantic_core._pydantic_core import list_all_errors

logging.getLogger("vbm_data_processing_logger").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class DataLoader:

    def __init__(self):
        pass

    async def upload_data_from_file(self, task):
        logger.info("saving data to temp zip file")

        await task_storage.update_task(task.task_id, status="UNZIPPING _DATA", progress=10)        
        folder = os.path.join(TEMP_FOLDER, task.task_id)
        os.mkdir(folder)

        logger.info("reading  data from zip file, unzipping")
        await self.get_data_from_zipfile(task.file_path, folder)

        zip_filename = os.path.basename(task.file_path)
        zip_filename_without_ext = os.path.splitext(zip_filename)[0]
        data_file_path = os.path.join(folder, f"{zip_filename_without_ext}.json")

        with open(data_file_path, 'r', encoding='utf-8-sig') as fp:
            json_data = json.load(fp)

        logger.info("validatind uploaded data")
        await task_storage.update_task(task.task_id, status="VALIDATING _DATA", progress=20)         
        data = data_validator.validate_raw_data(json_data)

        pd_data = pd.DataFrame(data) 
        pd_data['accounting_db'] = task.accounting_db

        data = pd_data.to_dict(orient='records')

        logger.info("writing data to db")
        await task_storage.update_task(task.task_id, status="WRITING _TO_DB", progress=60)          
        db_processor.set_accounting_db(task.accounting_db)
        if task.replace:
            await db_processor.delete_many('raw_data')

        await db_processor.insert_many('raw_data', data)

        await task_storage.cleanup_task_files(task.task_id)        

        return data

    async def get_data_from_zipfile(self, zip_file_path, folder):

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(folder)

    async def delete_data(self, accounting_db='', db_filter=None):
        db_processor.set_accounting_db(accounting_db)

        await db_processor.delete_many('raw_data',  db_filter=db_filter)

    async def get_data_count(self,  accounting_db='', db_filter=None):
        db_processor.set_accounting_db(accounting_db)

        result = await db_processor.get_count('raw_data',  db_filter=db_filter)
        return result


class DataValidator:

    def validate_raw_data(self, data):

        adapter = TypeAdapter(List[RawqDataStr])
        all_errors = defaultdict(list)
        results = []
        model_fields = set(RawqDataStr.model_fields.keys())

        for index, item in enumerate(data):
            try:
             
                extra_fields = {k: v for k, v in item.items() if k not in model_fields or k=='period'}

                self.validate_fields(extra_fields)

                results.append(extra_fields)
            except ValidationError as e:
                for error in e.errors():
                    error["ctx"] = error.get("ctx", {})
                    error["ctx"]["index"] = index
                    all_errors[index].append(error)
                results.append(None)  

        if all_errors:
            aggregated_errors = []
            for index, errors in all_errors.items():
                aggregated_errors.append({
                    "type": "value_error",
                    "loc": (index,),
                    "input": data[index],
                    "errors": errors,
                    "ctx": {"error": 'Fields are not allowed'}
                })
            
            raise ValidationError.from_exception_data(
                title="Validation errors in raw data",
                line_errors=aggregated_errors,
            )

        return results

    def validate_fields(self, fields_dict):

        errors = []
        for field, value in fields_dict.items():
            if field == 'period':
                fields_dict['period'] = datetime.strptime(value, '%d.%m.%Y')
            elif field.startswith('ind_'):
                field_parts = field.split('_') 
                if len(field_parts) != 2:
                    errors.append(InitErrorDetails({
                        "type": "value_error",
                        "loc": ("extra_fields",),
                        "msg": 'Field {} is not allowed'.format(field),
                        "input": value,
                        "error": ValueError('Field not allowed'),
                        "ctx": {"field": field, 
                                "error": 'Field {} is not allowed'.format(field)}
                    }))  
                if field_parts[1] not in ['name', 'id', 'kind']:
                    errors.append(InitErrorDetails({
                        "type": "value_error",
                        "loc": ("extra_fields",),
                        "msg": 'Field {} is not allowed'.format(field),
                        "input": value,
                        "error": ValueError('Field not allowed'),
                        "ctx": {"field": field, 
                                "error": 'Field {} is not allowed'.format(field)}
                    }))                                            
            else:
                field_parts = field.split('_')
                if len(field_parts) != 3:
                    errors.append(InitErrorDetails({
                        "type": "value_error",
                        "loc": ("extra_fields",),
                        "msg": 'Field {} is not allowed'.format(field),
                        "input": value,
                        "error": ValueError('Field not allowed'),
                        "ctx": {"field": field, 
                                "error": 'Field {} is not allowed'.format(field)}
                    }))
                
                if field_parts[0] not in ['dim', 'an', 'num']:
                    errors.append(InitErrorDetails({
                        "type": "value_error",
                        "loc": ("extra_fields",),
                        "msg": 'Field {} is not allowed'.format(field),
                        "input": value,
                        "error": ValueError('Field not allowed'),
                        "ctx": {"field": field, 
                                "error": 'Field {} is not allowed'.format(field)}
                    }))   

                if field_parts[0] in ['dim', 'an'] and field_parts[2] not in ['name', 'id', 'kind']:
                    errors.append(InitErrorDetails({
                        "type": "value_error",
                        "loc": ("extra_fields",),
                        "msg": 'Field {} is not allowed'.format(field),
                        "input": value,
                        "error": ValueError('Field not allowed'),
                        "ctx": {"field": field, 
                                "error": 'Field {} is not allowed'.format(field)}
                    }))
                elif field_parts[0] == 'num' and field_parts[2] not in ['value', 'kind']:
                    errors.append(InitErrorDetails({
                        "type": "value_error",
                        "loc": ("extra_fields",),
                        "msg": 'Field {} is not allowed'.format(field),
                        "input": value,
                        "error": ValueError('Field not allowed'),
                        "ctx": {"field": field, 
                                "error": 'Field {} is not allowed'.format(field)}
                    }))

                try:
                    f_ind = int(field_parts[1])
                except ValueError:
                    errors.append(InitErrorDetails({
                        "type": "value_error",
                        "loc": ("extra_fields",),
                        "msg": 'Field {} is not allowed'.format(field),
                        "input": value,
                        "error": ValueError('Field not allowed'),
                        "ctx": {"field": field, 
                                "error": 'Field {} is not allowed'.format(field)}
                    }))

        if errors:
            # exception_text = 'Field(s) {} is (are) not allowed! Raw data extra fields must start with ' \
            #     '"dim_", "an_" or "num_", then must be number (index) and finally must be "_id", "_kind" or "_name"'.format(', '.join(error_fields))
            raise ValidationError.from_exception_data(
                title="Validation errors in raw data",
                line_errors=errors,
            )


        return fields_dict


class Reader:
    
    async def read(self, data_filter):
        data = await db_processor.find("raw_data", convert_dates_in_db_filter(data_filter))
        pd_data = pd.DataFrame(data) 

        return pd_data


class Checker(BaseEstimator, TransformerMixin):

    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict

    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):

        if X.empty:
            raise ValueError('Fitting dataset is empty. Load more data or change filter.')
        
        return X


class RowToColumn(BaseEstimator, TransformerMixin):

    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict
        self.x_columns = []
        self.y_columns = []
        self.columns_descriptions = {}
        self.analytic_key_settings = {}
        self.only_outers = False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        indicators = [el['id'] for el in self.parameters['indicators']]
        data = X.loc[X['ind_id'].isin(indicators)].copy()
        data['analytic_key'] = data.apply(self._get_analytic_key_from_row, axis=1)
        result_data = self._group_data_with_dims(data)

        for indicator_ind, indicator in enumerate(indicators):

            indicator_settings = [el for el in self.parameters['indicators'] if el['id']==indicator][0]
            analytic_kinds = indicator_settings['analytics']

            if self.for_predict and indicator_settings['outer']:
                continue            

            if analytic_kinds:
                result_data = self._add_analytic_columns_to_data(result_data, data, indicator_settings, indicator_ind)
            else:
                result_data = self._add_ind_columns_to_data(result_data, data, indicator_settings, indicator_ind)
                
            if not self.for_predict:
                self.parameters['x_columns'] = self.x_columns
                self.parameters['y_columns'] = self.y_columns
                self.parameters['columns_descriptions'] = self.columns_descriptions

        return result_data
    
    def inverse_transform(self, X):

        non_model_columns = ['period'] + ['{}_id'.format(el) for el in self.parameters['dimensions']]

        in_data = X[non_model_columns].copy()
        in_data['ind_id'] = ''

        all_an_kind = []
        all_nums = []
        for ind in self.parameters['indicators']:
            if ind['analytics']:
                all_an_kind.extend(ind['analytics'])
            all_nums.extend(ind['numbers'])

        all_an_kind = list(set(all_an_kind))
        all_nums = list(set(all_nums))  

        for an_kind in all_an_kind:
            in_data['{}_id'.format(an_kind)] = ''
        
        for num in all_nums:
            in_data['{}_value'.format(num)] = 0

        data_dict = {}

        for column_name, column_settings in self.parameters['columns_descriptions'].items():
            if self.only_outers and not column_settings['outer']:
                continue
            portion_name = '{}_{}'.format(column_settings['indicator_id'], column_settings['analytic_key'])
            c_data = data_dict.get(portion_name)

            if c_data:
                c_data['{}_value'.format(column_settings['num'])] = X[column_name]
            else:
                c_data = in_data.copy()
                c_data['ind_id'] = column_settings['indicator_id']
                for an_name, an_value in column_settings['analytics'].items():
                    c_data['{}_id'.format(an_name)] = an_value    

                c_data['{}_value'.format(column_settings['num'])] = X[column_name]

                data_dict[portion_name] = c_data

        y = pd.concat(data_dict.values(), axis=0)

        return y
    
    def _add_ind_columns_to_data(self, result_data, initial_data, 
                            indicator_settings, 
                            indicator_ind):
        
        to_group = ['period'] + ['{}_id'.format(el) for el in self.parameters['dimensions']]
        columns = to_group + ['{}_value'.format(el) for el in indicator_settings['numbers']]
        ind_data = initial_data[columns].loc[initial_data['ind_id']==indicator_settings['id']].groupby(to_group, as_index=False).sum()
        num_to_rename = {}

        c_columns = []
        for num_name in indicator_settings['numbers']:
            column_name = self._get_column_name(indicator_ind, num_name)

            num_to_rename['{}_value'.format(num_name)] = column_name   

            if indicator_settings['outer']:
                self.y_columns.append(column_name)
            else:
                self.x_columns.append(column_name)

            c_columns.append(column_name)

            self.columns_descriptions[column_name] = {'indicator_id': indicator_settings['id'], 
                                                    'analytics': {},
                                                    'analytic_key': '',
                                                    'period_shift': 0,
                                                    'num': num_name,
                                                    'outer': indicator_settings['outer']}

        ind_data = ind_data.rename(num_to_rename, axis=1)

        if ind_data.empty:
            for col in c_columns:
                result_data[col] = 0
        else:
            to_merge = ['period'] + ['{}_id'.format(el) for el in self.parameters['dimensions']]
            result_data = result_data.merge(ind_data, on=to_merge, how='left')

        return result_data

    def _add_analytic_columns_to_data(self, result_data, initial_data, 
                            indicator_settings, 
                            indicator_ind):
        
        to_group = ['period'] + ['{}_id'.format(el) for el in self.parameters['dimensions']] + ['analytic_key']
        columns = to_group + ['{}_value'.format(el) for el in indicator_settings['numbers']]
        ind_data = initial_data[columns].loc[initial_data['ind_id']==indicator_settings['id']].groupby(to_group, as_index=False).sum() 

        an_keys = sorted(list(ind_data['analytic_key'].unique()))

        for an_ind, an_key in enumerate(an_keys):
            result_data = self._add_analytic_value_columns_to_data(result_data, ind_data, indicator_settings, indicator_ind, an_key, an_ind)
        
        return result_data
    
    def _add_analytic_value_columns_to_data(self, result_data, initial_data, indicator_settings, indicator_ind, analytic_key, analytic_ind):

        an_data = initial_data.loc[initial_data['analytic_key'] == analytic_key]

        to_group = ['period'] + ['{}_id'.format(el) for el in self.parameters['dimensions']]
        columns = to_group + ['{}_value'.format(el) for el in indicator_settings['numbers']]
   
        an_data = an_data[columns].groupby(to_group, as_index=False).sum() 
        num_to_rename = {}
        c_columns = []
        for num_name in indicator_settings['numbers']:
            column_name = self._get_column_name(indicator_ind, num_name, analytic_ind)

            num_to_rename['{}_value'.format(num_name)] = column_name   

            if indicator_settings['outer']:
                self.y_columns.append(column_name)
            else:
                self.x_columns.append(column_name)

            c_columns.append(column_name)
            

            self.columns_descriptions[column_name] = {'indicator_id': indicator_settings['id'], 
                                                    'analytics': self.analytic_key_settings[analytic_key],
                                                    'analytic_key': analytic_key,
                                                    'period_shift': 0,
                                                    'num': num_name,
                                                    'outer': indicator_settings['outer']}
        
        an_data = an_data.rename(num_to_rename, axis=1)

        if an_data.empty:
            for col in c_columns:
                result_data[col] = 0
        else:
            to_merge = ['period'] + ['{}_id'.format(el) for el in self.parameters['dimensions']]
            result_data = result_data.merge(an_data, on=to_merge, how='left')

        return result_data

    def _group_data_with_dims(self, data):
        to_group = ['period'] + ['{}_id'.format(el) for el in self.parameters['dimensions']]
        grouped_data = data[to_group].groupby(to_group, as_index=False).sum()
        return grouped_data 
    
    def _get_analytic_key_from_row(self, row):
        indicator = row['ind_id']
        analytic_kinds = [el['analytics'] for el in self.parameters['indicators'] if el['id']==indicator][0]

        vv = []
        an_settings = {}

        if analytic_kinds:

            for an in analytic_kinds:
                vv.append(an)
                vv.append(row['{}_id'.format(an)])
                an_settings[an] = row['{}_id'.format(an)]

            str_v = '_'.join(vv)
            result = str(uuid.uuid3(uuid.NAMESPACE_DNS, str_v))
        else:
            result = ''

        if result not in self.analytic_key_settings:
            self.analytic_key_settings[result] = an_settings

        return result

    def _get_column_name(self, indicator_ind, num_name, analytic_ind=None):
        if analytic_ind is not None:
            return 'ind_{}_an_{}_{}'.format(indicator_ind, analytic_ind, num_name)
        else:
            return 'ind_{}_{}'.format(indicator_ind, num_name)


class NanProcessor(BaseEstimator, TransformerMixin):
    """ Transformer for working with nan values (deletes nan rows, columns, fills 0 to na values) """

    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict


    def fit(self, X, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Process nan values: removes all nan rows and columns, fills 0 instead single nan values
        :param x: data before nan processing
        :return: data after na  processing
        """

        x = x.fillna(0)

        return x


class Shuffler(BaseEstimator, TransformerMixin):
    """
    Transformer class to shuffle data rows
    """
    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict        

    def fit(self, X, y=None):
        return self
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.sample(frac=1).reset_index(drop=True).copy()


def convert_dates_in_db_filter(db_filter, is_period=False):
    if isinstance(db_filter, list):
        result = []
        for el in db_filter:
            result.append(convert_dates_in_db_filter(el, is_period))
    elif isinstance(db_filter, dict):
        result = {}
        for k, v in db_filter.items():
            if k=='period':
                result[k] = convert_dates_in_db_filter(v, True)        
            else:
                result[k] = convert_dates_in_db_filter(v, is_period)
    elif isinstance(db_filter, str) and is_period:
        result = datetime.strptime(db_filter, '%d.%m.%Y')
    else:
        result = db_filter

    return result


data_loader = DataLoader()
data_validator = DataValidator()
