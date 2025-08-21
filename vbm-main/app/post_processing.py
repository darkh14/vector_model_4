from db import db_processor
import shap
import logging
from data_processing import data_procesor
from entities import FIStatuses, ModelStatuses
from errors import AlsoCalculatingException
import pandas as pd
import numpy as np
from typing import Optional, Any

import plotly.graph_objects as go
import plotly.express as px

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

    async def get_sa(self, model, X, x_indicators, y_indicator, coefs, auto_selection_number, get_graph):
        if model.status != ModelStatuses.READY:
            raise ValueError('Model is not ready calculate sensitivity analysis. Fit model before!')
         
        for x_ind in x_indicators:
            if x_ind not in [el['id'] for el in model.parameters['indicators'] if not el['outer']]:
                raise ValueError('Indicator "{}" is not in x indicators'.format(x_ind))

        if y_indicator not in [el['id'] for el in model.parameters['indicators'] if el['outer']]:
            raise ValueError('Indicator "{}" is not in y indicators'.format(y_indicator))      

        dim_coef_settings = self.get_dim_coef_settings(X, x_indicators, coefs, model.parameters['indicators'])
        dataset_y_init = pd.DataFrame(X)
        dataset_y_init = dataset_y_init.loc[dataset_y_init['ind_id'] == y_indicator]
        dataset_y_init['value'] = dataset_y_init.apply(lambda row: self._get_sa_value(row, model.parameters['indicators']), axis=1) 
        y_0_value = dataset_y_init['value'].sum()     

        dataset = await self._get_dataset_for_sa(X, dim_coef_settings, model.parameters['indicators'])

        result_dataset = await model.predict(dataset.to_dict(orient='records'))
        result_dataset = pd.DataFrame(result_dataset)
        result_dataset = result_dataset.loc[result_dataset['ind_id'] == y_indicator].copy()

        result_dataset['coef'] = result_dataset.apply(lambda row: self._get_sa_coef(row, dim_coef_settings), axis=1)
        result_dataset['sa_ind_id'] = result_dataset.apply(lambda row: self._get_sa_indicator(row, dim_coef_settings), axis=1)
        result_dataset['sa_ind_kind'] = result_dataset.apply(lambda row: self._get_sa_indicator_kind(row, dim_coef_settings), axis=1)        
        result_dataset['value'] = result_dataset.apply(lambda row: self._get_sa_value(row, model.parameters['indicators']), axis=1)                   

        result_dataset = result_dataset.groupby(['sa_ind_id', 'sa_ind_kind', 'coef'], as_index=False).agg({'value': 'sum'})
        result_dataset = result_dataset.rename({'sa_ind_id': 'ind_id', 'sa_ind_kind': 'ind_kind'}, axis=1)

        result_dataset['value_0'] = y_0_value
        result_dataset['delta'] = result_dataset['value'] - result_dataset['value_0']
        result_dataset['relative_delta'] = result_dataset.apply(lambda row: row['delta']/row['value_0'] if row['value_0'] else 0, axis=1)    

        result_dataset = self._trunc_sa_output_data(result_dataset, auto_selection_number)

        if get_graph:
            graph_html = graph_processor.get_sa_graph(result_dataset, model.parameters['indicators'])
        else:
            graph_html = ''

        return result_dataset.to_dict(orient='records'), graph_html
    
    async def get_fa(self, model, dataset, scenarios, x_indicators: list[str],
                            y_indicator: str,
                            get_graph: bool = False):
        
        if model.status != ModelStatuses.READY:
            raise ValueError('Model is not ready calculate sensitivity analysis. Fit model before!')
         
        for x_ind in x_indicators:
            if x_ind not in [el['id'] for el in model.parameters['indicators'] if not el['outer']]:
                raise ValueError('Indicator "{}" is not in x indicators'.format(x_ind))

        if y_indicator not in [el['id'] for el in model.parameters['indicators'] if el['outer']]:
            raise ValueError('Indicator "{}" is not in y indicators'.format(y_indicator))          

        dataset = pd.DataFrame(dataset)
        dataset_base = dataset.loc[dataset['dim_0_id'] == scenarios['base']['id']].copy()
        dataset_calculated = dataset.loc[dataset['dim_0_id'] == scenarios['calculated']['id']].copy()

        used_indicator_ids = []

        dataset_list = []

        part_dataset = dataset_base.copy()  
        part_dataset['dim_0_id'] = 'base' 
        dataset_list.append(part_dataset) 

        sc_list = ['base']   

        for idx, input_ind in enumerate(x_indicators):

            part_dataset = self._sa_combine_dataset_from_base_calculated(dataset_base, dataset_calculated, used_indicator_ids, input_ind)
            part_dataset['dim_0_id'] = input_ind
            sc_list.append(input_ind)
            dataset_list.append(part_dataset)

            used_indicator_ids.append(input_ind)

        dataset = pd.concat(dataset_list, axis=0)
        sc_dict = dict(zip(sc_list, range(len(sc_list))))
        y_dataset = await model.predict(dataset.to_dict(orient='records'))
        y_dataset = pd.DataFrame(y_dataset)
        y_dataset['to_sort'] = y_dataset['dim_0_id'].apply(lambda x: sc_dict[x])

        y_dataset = y_dataset.sort_values('to_sort')
        y_dataset = y_dataset.loc[y_dataset['ind_id'] == y_indicator]

        y_dataset['value_1'] = 0
        
        y_ind_descr = [el for el in model.parameters['indicators'] if el['id'] == y_indicator][0]
        for num in y_ind_descr['numbers']:
            y_dataset['value_1'] = y_dataset['value_1'] + y_dataset['{}_value'.format(num)]

        y_dataset = y_dataset[['dim_0_id', 'value_1', 'to_sort']].groupby(['dim_0_id', 'to_sort'], as_index=False).sum('value_1')
        y_dataset = y_dataset.sort_values('to_sort').reset_index(drop=True)
        y_dataset_2 = y_dataset.copy()
        y_dataset_2 = y_dataset_2.rename({'value_1': 'value_0'}, axis=1)
        y_dataset_2 = y_dataset_2[['to_sort', 'value_0']]
        y_dataset_2['to_sort'] = y_dataset_2['to_sort'] + 1

        y_dataset_3 = y_dataset.loc[y_dataset['dim_0_id'] == x_indicators[-1]].copy()
        y_dataset_3['dim_0_id'] = 'calculated'

        y_dataset = pd.concat([y_dataset, y_dataset_3], axis=0)

        y_dataset = y_dataset.merge(y_dataset_2, on='to_sort', how='left').fillna(0)
        y_dataset['value'] = y_dataset['value_1'] - y_dataset['value_0']
        y_dataset.loc[y_dataset['dim_0_id'] == 'calculated', 'value'] = scenarios['calculated']['value']

        y_dataset = y_dataset.drop('to_sort', axis=1)
        y_dataset = y_dataset.rename({'dim_0_id': 'ind_id'}, axis=1)
        y_dataset['ind_kind'] = y_dataset['ind_id'].apply(lambda x: self._get_fa_ind_kind(x, model.parameters['indicators']))
        
        if get_graph:
            graph_data = graph_processor.prepare_data_for_fa_graph(model, y_dataset, scenarios)
            graph_html = graph_processor.get_fa_graph(model, graph_data, y_indicator)
        else:
            graph_html = ''


        return y_dataset.to_dict(orient='records'), graph_html      

    def _get_fa_ind_kind(self, value, indicators):
        if value in ['base', 'calculated']:
            return ''
        else:
            return [el['kind'] for el in indicators if el['id'] == value][0]
        

    def _sa_combine_dataset_from_base_calculated(self, dataset_base, dataset_calculated, used_indicators, current_ind=''):
        
        calc_indicators = used_indicators
        if current_ind:
            calc_indicators.append(current_ind)

        part_1_dataset = dataset_base.loc[~dataset_base['ind_id'].isin(calc_indicators)].copy()
        part_2_dataset = dataset_calculated.loc[dataset_calculated['ind_id'].isin(calc_indicators)].copy()

        return pd.concat([part_1_dataset, part_2_dataset], axis=0)

    def _get_sa_coef(self, row, dim_coef_settings):
        result = dim_coef_settings[row['dim_0_id']]['coef']

        return result

    def _get_sa_indicator(self, row, dim_coef_settings):
        cof_settings = dim_coef_settings.get(row['dim_0_id'])
        if cof_settings:
            result = cof_settings['indicator']
        else:
            result = row['ind_id']

        return result
    
    def _get_sa_indicator_kind(self, row, dim_coef_settings):
        coef_settings = dim_coef_settings.get(row['dim_0_id'])
        if coef_settings:
            result = coef_settings['ind_kind']
        else:
            result = row['ind_kind']

        return result    

    def _get_sa_value(self, row, indicators_settings):
        result = 0
        for num in [el for el in indicators_settings if el['id']==row['ind_id']][0]['numbers']:
            result += row['{}_value'.format(num)]
        return result    

    async def _get_dataset_for_sa(self, X, dim_coef_settings, indicators_settings):
        init_dataset = pd.DataFrame(X)

        dataset_list = []
        
        indicators = list(set([el['indicator'] for el in dim_coef_settings.values()]))
        coefs = list(set([el['coef'] for el in dim_coef_settings.values()]))

        for ind in indicators:
            ind_settings = [el for el in indicators_settings if el['id'] == ind][0]
            for coef in coefs:                
                c_dataset = init_dataset.copy()
                
                for col in ind_settings['numbers']:
                    c_dataset['{}_value'.format(col)] = c_dataset.apply(lambda row: self._set_sa_deviation_value(row, col, ind, coef), axis=1)
                dim_col = [k for k, v in dim_coef_settings.items() if v['indicator'] == ind and v['coef'] == coef][0]
                c_dataset['dim_0_id'] = dim_col

                dataset_list.append(c_dataset)

        result_dataset = pd.concat(dataset_list, axis=0).reset_index(drop=True)

        return result_dataset
    
    def _set_sa_deviation_value(self, row, col, ind, coef):
        if row['ind_id'] == ind:
            return row['{}_value'.format(col)]*(1+coef)
        else:
            return row['{}_value'.format(col)]

    def get_dim_coef_settings(self, dataset, indicators, coefs, indicators_descr):

        dims = list(pd.DataFrame(dataset)['dim_0_id'].unique())
        result_dict = {}
        if 0 not in coefs:
            coefs.append(0)

        for dim in dims:
            for ind in indicators:
                ind_descr = [el for el in indicators_descr if el['id'] == ind][0]
                for coef in coefs:
                    if coef:
                        scenario_new_1 = '{}_{}_{}'.format(dim, ind, coef)
                        result_dict[scenario_new_1] = {'dim_0': dim, 'indicator': ind, 'ind_kind': ind_descr['kind'], 'coef': coef}

                        scenario_new_2 = '{}_{}_{}'.format(dim, ind, -coef)
                        result_dict[scenario_new_2] = {'dim_0': dim, 'indicator': ind, 'ind_kind': ind_descr['kind'], 'coef': -coef}
                    else:
                        scenario_new = '{}_{}_{}'.format(dim, ind, coef)
                        result_dict[scenario_new] = {'dim_0': dim, 'indicator': ind, 'ind_kind': ind_descr['kind'], 'coef': coef}


        return result_dict

    def _trunc_sa_output_data(self, dataset: pd.DataFrame, auto_selection_number):

        grouped_dataset = dataset.copy()
        grouped_dataset['abs_value'] = grouped_dataset['value'].apply(abs)
        grouped_dataset = grouped_dataset[['ind_id', 'abs_value']].groupby(['ind_id'], as_index=False).sum('abs_value')
        grouped_dataset = grouped_dataset.sort_values('abs_value').reset_index(drop=True)
        grouped_dataset['sort_inds'] = grouped_dataset.index

        dataset = dataset.merge(grouped_dataset[['ind_id', 'sort_inds']], on=['ind_id'])

        dataset = dataset.sort_values(['sort_inds', 'coef']).reset_index(drop=True)
        if auto_selection_number:
            dataset = dataset.loc[dataset['sort_inds'] < auto_selection_number].copy()

        dataset = dataset.drop('sort_inds', axis=1)

        return dataset


class GraphProcessor:
    def __init__(self):
        pass

    def get_sa_graph(self, dataset: pd.DataFrame, indicators_descr) -> str:
        """
        Forms sensitivity analysis graph html
        @param graph_data: sensitivity analysis data
        @return: string of html graph
        """

        colors_dict = self._get_colors_for_data(dataset)

        x = [el*100 for el in dataset['coef'].unique()]
        x.append(0)

        x.sort()

        y_list = []
        indicator_names = []

        indicators = list(dataset['ind_id'].unique())

        for ind_id in indicators:

            c_ind_names = [el['name'] for el in indicators_descr if el['id'] == ind_id]

            indicator_names.append(c_ind_names[0] if c_ind_names else ind_id)

            element_data = dataset.loc[dataset['ind_id'] == ind_id].copy()

            element_data_0 = element_data.iloc[[0]].copy()
            element_data_0['coef'] = 0
            element_data_0['value'] = element_data_0['value_0']
            element_data_0['delta'] = 0
            element_data_0['relative_delta'] = 0

            element_data = pd.concat((element_data, element_data_0), axis=0)

            element_data.sort_values('coef', inplace=True)

            y_c = list(element_data['relative_delta'].to_numpy())
            y_c = list(map(lambda z: 100*z, y_c))

            y_list.append(y_c)

        fig = go.Figure()

        colors = self._get_plotly_colors()

        for ind, y in enumerate(y_list):
            fig.add_trace(go.Scatter(x=x, y=y, name=indicator_names[ind],
                                     line=dict(color=colors_dict[indicators[ind]])))

        font_size = 10

        fig.update_layout(title=dict(text='Анализ на чувствительность', font=dict(size=font_size+1)), showlegend=False,
                          xaxis_title=dict(text="Отклонения входного показателя, %", font=dict(size=font_size)),
                          yaxis_title=dict(text="Отклонения выходного показателя, %", font=dict(size=font_size)),
                          paper_bgcolor='White',
                          plot_bgcolor='White')

        fig.update_layout(legend=dict(
                                    x=0,
                                    y=-0.2,
                                    traceorder="normal",
                                    orientation='h',
                                    font=dict(size=font_size),
                                ))

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', zerolinecolor='Grey', tickvals=x)

        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', zerolinecolor='Grey')

        graph_html = fig.to_html()

        return graph_html

    def prepare_data_for_fa_graph(self, model, result_data: pd.DataFrame, 
                                  scenarios) -> pd.DataFrame:

        result_data = result_data.loc[~result_data['ind_id'].isin(['base', 'calculated'])].copy()
        all_indicators = [el for el in result_data['ind_id'].unique() if el != 'base']

        indicators_dict = {el['id']: el['name'] for el in model.parameters['indicators'] if el['id'] in all_indicators}
        indicators_dict['base'] = 'Базовый'

        result_data['title'] = result_data['ind_id'].apply(lambda x: indicators_dict[x])

        need_to_add_other_line = False# len(result_data['indicator'].unique()) < len(self.parameters.x_indicators)
        result_data['order'] = list(range(2, result_data.shape[0]+2))

        result_data.drop(['ind_id', 'ind_kind'], axis=1, inplace=True)

        base_line = {'title': scenarios['base']['name'], 'value': scenarios['base']['value'], 'order': 1}

        lines_to_add = [base_line]

        order_of_calculated = result_data.shape[0] + 2
        if need_to_add_other_line:

            sum_all = float(result_data[['value']].apply(sum, axis=0))
            other_value = scenarios['calculated']['value'] - sum_all - scenarios['base']['value']

            if abs(other_value) >= 10:

                other_line = {'title': 'Прочие факторы', 'value': other_value, 'order': order_of_calculated}
                order_of_calculated += 1

                lines_to_add.append(other_line)

        calculated_line = {'title': scenarios['calculated']['name'],
                           'value': scenarios['calculated']['value'],
                           'order': order_of_calculated}

        lines_to_add.append(calculated_line)

        result_data = pd.concat([result_data, pd.DataFrame(lines_to_add)])

        result_data = result_data.sort_values('order')

        return result_data

    def get_fa_graph(self, model, graph_data: pd.DataFrame, y_indicator):
        output_indicator_descr = [el for el in model.parameters['indicators'] if el['id'] == y_indicator][0]

        x_list = list(graph_data['title'])

        x_list_fa = [el[:30] + '...' if len(el) > 30 else el for el in x_list]
        y_list = list(graph_data['value'])

        text_list = []
        hover_text_list = []

        initial_value = 0
        for index, item in enumerate(y_list):
            if item > 0 and index != 0 and index != len(y_list) - 1:
                text_list.append('+{0:,.0f}'.format(y_list[index]).replace(',', ' '))
            else:
                text_list.append('{0:,.0f}'.format(y_list[index]).replace(',', ' '))

            hover_value = '{}<br>'.format(x_list[index]) + '{0:,.0f}'.format(item).replace(',', ' ')

            if index in (0, len(y_list)-1):
                hover_text_list.append('{}'.format(hover_value))
            else:
                if item > 0:
                    hover_value += ' &#9650;'
                elif item < 0:
                    hover_value += ' &#9660;'

                hover_text_list.append('{}<br>Предыдущее: {}'.format(hover_value,
                                                                '{0:,.0f}'.format(initial_value).replace(',', ' ')))

            initial_value += item

        for index, item in enumerate(text_list):
            if item[0] == '+' and index != 0 and index != len(text_list) - 1:
                text_list[index] = '<span style="color:#2ca02c">' + text_list[index] + '</span>'
            elif item[0] == '-' and index != 0 and index != len(text_list) - 1:
                text_list[index] = '<span style="color:#d62728">' + text_list[index] + '</span>'
            if index == 0 or index == len(text_list) - 1:
                text_list[index] = '<b>' + text_list[index] + '</b>'

        dict_list = []
        for i in range(0, 1200, 200):
            dict_list.append(dict(
                type="line",
                line=dict(
                    color="#666666",
                    dash="dot"
                ),
                x0=-0.5,
                y0=i,
                x1=6,
                y1=i,
                line_width=1,
                layer="below"))

        fig = go.Figure(go.Waterfall(
            name="Factor analysis", orientation="v",
            measure=["absolute", *(graph_data.shape[0]-2) * ["relative"], "total"],
            x=x_list_fa,
            y=y_list,
            text=text_list,
            textposition="outside",
            connector={"line": {"color": 'rgba(0,0,0,0)'}},
            increasing={"marker": {"color": "#2ca02c"}},
            decreasing={"marker": {"color": "#d62728"}},
            totals={'marker': {"color": "#9467bd"}},
            textfont={"family": "Open Sans, light",
                      "color": "black"
                      }
        ))

        fig.update_layout(
            title={'text': '<b>Факторный анализ</b><br><span style="color:#666666">{}</span>'.format(
                output_indicator_descr['name'])},
            showlegend=False,
            height=650,
            font={
                'family': 'Open Sans, light',
                'color': 'black',
                'size': 14
            },
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickformat=",.0f"),
            yaxis_title="руб.",
            shapes=dict_list
        )

        fig.update_xaxes(tickangle=-45, tickfont=dict(family='Open Sans, light', color='black', size=14))

        y_tick_vals, y_tick_texts = self._get_y_vals_texts_for_fa_graph(y_list)

        fig.update_yaxes(tickangle=0, tickfont=dict(family='Open Sans, light', color='black', size=14),
                         tickvals=y_tick_vals, ticktext=y_tick_texts)

        fig.update_traces(hoverinfo='text', hovertext=hover_text_list)

        graph_str = fig.to_html()

        return graph_str

    def _get_colors_for_data(self, graph_data: pd.DataFrame):
        colors = self._get_plotly_colors()
        indicators = list(graph_data['ind_id'].unique())

        colors_dict = {}
        for idx, indicator in enumerate(indicators):
            color_idx = idx
            while True:
                if color_idx < len(colors):
                    break
                else:
                    color_idx -= len(colors)
            colors_dict[indicator] = colors[idx]

        return colors_dict
    
    @staticmethod
    def _get_plotly_colors():
        return (px.colors.qualitative.Plotly +
                px.colors.qualitative.Alphabet +
                px.colors.qualitative.Dark24 +
                px.colors.qualitative.Light24)
    
        
    @staticmethod
    def _get_y_vals_texts_for_fa_graph(y_values: list[int | float]):
        """
        Returns text of y-axis of fa graph
        @param y_values: values of y-axis
        @return: list of y-texts
        """
        max_value = 0
        current_value = 0

        for index, y_value in enumerate(y_values):
            if index == 0:
                current_value = 0
                max_value = y_value
            elif index == len(y_values):
                current_value = y_value
            else:
                current_value += y_value

            if current_value > max_value:
                max_value = current_value

        max_value = 1.5*max_value

        step = max_value/10

        step_pow = 0
        c_step = step

        while c_step > 10:
            c_step = c_step // 10
            step_pow += 1

        step = float('5e{}'.format(step_pow))
        step = int(step)

        value = 0

        result_values = []
        result_texts = []

        while value < max_value:
            result_values.append(value)
            result_texts.append('{0:,.0f}'.format(value).replace(',', ' '))
            value += step

        return result_values, result_texts


post_processor = PostProcessor()
graph_processor = GraphProcessor()
