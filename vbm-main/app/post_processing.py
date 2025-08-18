from db import db_processor
import shap
import logging
from data_processing import data_procesor
from entities import FIStatuses, ModelStatuses
from errors import AlsoCalculatingException
import pandas as pd
import numpy as np
from typing import Optional

logging.getLogger("vbm_data_processing_logger").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class PostProcessor:

    def __init__(self):
        pass


    async def calculate_feature_importances(self, model, data_filter: Optional[dict] = None):
        
        try:
            if model.status == FIStatuses.CALCULATING:
                raise AlsoCalculatingException('Feature importances is calculating now!')
            
            if model.status  != ModelStatuses.READY:
                raise ValueError('Model is not ready to calculate fi!')      
            
            if model.fi_status == FIStatuses.CALCULATING:
                raise ValueError('Feature importances is already calculating!')                 

            model.fi_status = FIStatuses.CALCULATING
            model.fi_error_text = ''

            model.feature_importances = {}         

            logger.info("Calculating fi model. Model id={}".format(model.model_id))
            model.feature_importances = await self._get_fi_calculated(model, data_filter)

            model.fi_status = FIStatuses.CALCULATED
            logger.info("Saving model to db. Мodel id={}".format(model.model_id))
            await model.write_to_db()            
            logger.info("Calculating fi is finished. Model id={}".format(model.model_id))
            
        except AlsoCalculatingException as e:
            raise e
        except Exception as e:
            model.fi_status = FIStatuses.ERROR
            model.fi_error_text = str(e)

            await model.write_to_db()
            raise e            

        return True


    async def _get_fi_calculated(self, model, data_filter):

        def f(X):
            return model.ml_model.predict(X).flatten() 
        
        X_y = await data_procesor.get_dataset(model.model_id, model.scaler, model.parameters, data_filter)
        X = X_y[model.parameters['x_columns']]

        idx = min(X.shape[0], 30) 
        kernel_explainer = shap.KernelExplainer(f, X.iloc[:idx, :])
        shap_values = kernel_explainer.shap_values(X.iloc[:idx, :], nsamples=500)

        feature_names = model.parameters['x_columns']

        rf_resultX = pd.DataFrame(shap_values, columns = feature_names)

        vals = np.abs(rf_resultX.values).mean(0)

        shap_importance = dict(zip(feature_names, vals))

        sum_all = sum(list(shap_importance.values()))
        shap_importance = {k: v/sum_all if sum_all else 0 for k, v in shap_importance.items()}

        return shap_importance
    

    async def drop_fi(self, model):
        model.feature_importances = {}
        model.fi_status = FIStatuses.NOT_CALCULATED
        model.fi_error_text = ''

        await model.write_to_db()


    async def get_feature_importances(self, model):
        if model.fi_status != FIStatuses.CALCULATED:
            raise ValueError('Feature importances are not calculated! Calculate FI before')
        return {'fi': model.feature_importances, 'descr': model.parameters['columns_descriptions']}


    async def get_sa(self, model, X, indicators, koefs):
        pass

post_processor = PostProcessor()
