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

from pydantic_core._pydantic_core import list_all_errors

logging.getLogger("vbm_data_processing_logger").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class DataLoader:

    def __init__(self):
        pass

    async def upload_data_from_file(self, task):
        logger.info("saving data to temp zip file")
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
        data = data_validator.validate_raw_data(json_data)

        pd_data = pd.DataFrame(data) 
        pd_data['accounting_db'] = task.accounting_db

        data = pd_data.to_dict(orient='records')

        logger.info("writing data to db")
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


class Reader(BaseEstimator, TransformerMixin):
    
    async def read(self, data_filter):
        data = await db_processor.find("raw_data", data_filter)
        pd_data = pd.DataFrame(data) 

        return pd_data


class RowToColumn(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def get_data_for_fit(self, data, parameters):
        pass
        # pd_data = 



data_loader = DataLoader()
data_validator = DataValidator()
