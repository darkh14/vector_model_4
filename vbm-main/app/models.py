from db import db_processor
from typing import Optional
from sklearn.pipeline import Pipeline
from data_processing import Reader, RowToColumn


class Model:

    def __init__(self, model_id, parameters=None):
        self.model_id = model_id

        self.parameters = parameters
        self.data_filter = None

    async def initialize(self):
        if not self.parameters:
            self.parameters = await self.read_model(self.model_id)
        self.data_filter = self.parameters.data_filter


    @staticmethod
    async def read_model(model_id):
        return db_processor.find_one('models', {'model_id': model_id})
    
    @staticmethod
    async def write_model(model_id, model_data):
        return db_processor.insert_one('models', model_data, {'model_id': model_id})
    
    async def fit(self, data_filter: Optional[dict] = None):
        
        X_y = await Reader().read(self.data_filter)

        pipeline = Pipeline([('row_column_transformer', RowToColumn(self.parameters.data_filter))])

        # pipeline.fit_transform([], [])
        aa = 1

