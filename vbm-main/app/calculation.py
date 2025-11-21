
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import json
import math
import hashlib
    

class ProcessorRec:

    def __init__(self):
        pass    

    def initialize_fields(self, dataset):
        for q in [1, 2]:
            dataset['flats_pc_q{}'.format(q)] = 0
            dataset['flats_pc_q{}_acc'.format(q)] = 0
            dataset['flats_m2_q{}'.format(q)] = 0
            dataset['flats_rubm2_q{}'.format(q)] = 0 
            dataset['income_flats_q{}_past1'.format(q)] = 0

            dataset['income_flats_q{}'.format(q)] = 0
            dataset['flats_pc_q{}_all_qty'.format(q)] = 0 
            
            dataset['nonresidental_m2_q{}'.format(q)] = 0
            dataset['nonresidental_rubm2_q{}'.format(q)] = 0
            dataset['income_nonresidental_q{}'.format(q)] = 0

            dataset['parking_pc_q{}'.format(q)] = 0
            dataset['parking_pc_acc_q{}'.format(q)] = 0                   
            dataset['parking_rubpc_q{}'.format(q)] = 0
            dataset['income_parking_q{}'.format(q)] = 0 

            dataset['receipts_flats_q{}'.format(q)] = 0
            dataset['receipts_nonresidental_q{}'.format(q)] = 0   
            dataset['receipts_parking_q{}'.format(q)] = 0   

            dataset['revenue_after_q{}'.format(q)] = 0 
            
            dataset['cash_escrow_acc_q{}'.format(q)] = 0 
            dataset['disclosure_escrow_q{}'.format(q)] = 0 

            dataset['advertising_q{}'.format(q)] = 0
            dataset['commercial_staff_q{}'.format(q)] = 0
            dataset['remuneration_to_banks_q{}'.format(q)] = 0  
            dataset['adm_pers_q{}'.format(q)] = 0
            dataset['rent_q{}'.format(q)] = 0                 
            dataset['land_tax_q{}'.format(q)] = 0 

            dataset['escrow_cash_acc_q{}'.format(q)] = 0
            dataset['escrow_disclosure_q{}'.format(q)] = 0 

            dataset['smr_flats_q{}'.format(q)] = 0    
            dataset['escrow_repayment_c_q{}'.format(q)] = 0   
            dataset['escrow_repayment_percent_c_q{}'.format(q)] = 0  

            dataset['escrow_accural_percent_q{}'.format(q)] = 0 
            dataset['escrow_repayment_q{}'.format(q)] = 0             
                             
        dataset['revenue_after'] = 0
        dataset['escrow_disclosure'] = 0
        dataset['bridge_md_payout_sum_acc'] = 0  
        dataset['escrow_accural_percent'] = 0
        dataset['escrow_repayment'] = 0  

        dataset['vat_nonresidental_sum'] = 0
        dataset['vat_parking_sum'] = 0
        dataset['cost_price_sum'] = 0 
        dataset['bridge_payout_sum'] = 0 
        dataset['escrow_payout_sum'] = 0
        dataset['entrance_rights_sum'] = 0
        dataset['escrow_disclosure_sum'] = 0        
        dataset['revenue_after_sum'] = 0         
                                
    def get_flats_pc(self, row, row_prev=None, q=1):
        # 28-29
        # temp_build = 40 if q==1 else 32
        # temp_build_another = 32
        # temp_build_all = 40

        # value_all = 491 if q==1 else 1288
        # value_all_another = 1288
        
        temp_build = row['temp_build_q1'] if q==1 else row['temp_build_q2']
        
        temp_build_another = row['temp_build_q2']
        temp_build_all = 40

        value_all = 491 if q==1 else 1288
        value_all_another = 1288

        start_date_col = 'sale_start_period_q{}'.format(q)

        construction_period_col = 'construction_period_q{}'.format(q)
        
        result_another = 0        

        result = 0

        if row['period'] < row[start_date_col]:
            result = 0    
        else:        
            sum_prev = row_prev['flats_pc_q{}_acc'.format(q)] if row_prev is not None else 0
            rest = value_all - sum_prev

            if q==1:
                result_another = self.get_flats_pc(row, row_prev, q=2) # 96
            result = min(temp_build*3, rest)
            if result < 0:
                result = 0
        
        if q == 1:
            result = min(result, temp_build_all*3 - result_another) # 120-96 

        return result
    
    def get_flats_pc_acc(self, row, row_prev=None, q=1):
        result = self.get_flats_pc(row, row_prev, q)
        if row_prev is not None:
            result += row_prev['flats_pc_q{}_acc'.format(q)]
        return result
    
    def get_flats_m2(self, row, row_prev=None, q=1):
        # 39-40
        # flats_pc_col = 'flats_pc_q{}'.format(q)
        # if past:
        #     flats_pc_col = flats_pc_col + '_past{}'.format(past)
        flats_pc_all = 491 if q==1 else 1288
        flats_m2_all = 22087.8922 if q==1 else 57973.0078
        flats_pc = row['flats_pc_q{}'.format(q)]

        result = flats_pc*flats_m2_all/flats_pc_all

        return result

    def get_flats_rubm2(self, row, row_prev=None, q=1):
        # 50-51
        start_price = row['start_price_flats_q1'] if q==1 else row['start_price_flats_q2']
        start_period_col = 'sale_start_period_q{}'.format(q)

        construction_period = 12 if q==1 else 8
        price_grows_col = 'price_grows_q{}'.format(q)

        flats_pc = row['flats_pc_q{}'.format(q)]
        result = 0

        period = row['period']
        periods = 0

        if period < row[start_period_col] or flats_pc == 0:
            result = 0    
        elif period == row[start_period_col] and flats_pc > 0:
            result = start_price
        else:
            # periods = min(construction_period-1, period - row[start_period_col])
            price_grows = ((row[price_grows_col] + 1)**(1/(construction_period-1))-1)
            if row['period'] >= row[start_period_col] + construction_period:
                price_grows = 0
            # result = start_price*(1+price_grows)**periods  
            result = (row_prev['flats_rubm2_q{}'.format(q)] if row_prev else 0)*(1+price_grows)

        return result

    def get_income_flats(self, row, row_prev=None, q=1):
        # 109-110
        flats_rubm2 = self.get_flats_rubm2(row, row_prev, q)
        flats_m2 = self.get_flats_m2(row, row_prev, q)

        result = flats_m2*flats_rubm2/1000
      
        return result

    def get_income_flats_acc(self, row, row_prev=None, q=1):
        result = row['income_flats_q{}'.format(q)]
        if row_prev is not None:
            result += row_prev['income_flats_acc_q{}'.format(q)]
        
        return result

    def get_nonresidental_m2(self, row, row_prev=None, q=1):
        # 62-63
        nonresidental_pc_all = row['flats_pc_q{}_all_qty'.format(q)]
        nonresidental_m2_all = 2000 if q==1 else 2000

        nonresidental_m2 = (nonresidental_m2_all/nonresidental_pc_all) if nonresidental_pc_all else 0

        flats_pc = row['flats_pc_q{}'.format(q)]

        result = nonresidental_m2 if flats_pc else 0

        return result   

    def get_nonresidental_rubm2(self, row, row_prev=None, q=1):
        # 73-74
        start_price_nonresidental = row['start_price_nonresidental_q{}'.format(q)]
        flats_pc = row['flats_pc_q{}'.format(q)]
        result = 0
        if flats_pc:
            result = start_price_nonresidental
        return result

    def get_income_nonresidental(self, row, row_prev=None, q=1):
        # 120-121
        nonresidental_rubm2 = self.get_nonresidental_rubm2(row, row_prev, q)
        nonresidental_m2 = self.get_nonresidental_m2(row, row_prev, q)

        result = nonresidental_rubm2*nonresidental_m2/1000
     
        return result

    def get_income_nonresidental_acc(self, row, row_prev=None, q=1):
        result = row['income_nonresidental_q{}'.format(q)]
        if row_prev is not None:
            result += row_prev['income_nonresidental_acc_q{}'.format(q)]
        
        return result

    def get_parking_pc_acc(self, row, row_prev=None, q=1):
        result = row['parking_pc_q{}'.format(q)]
        if row_prev is not None:
            result += row_prev['parking_pc_acc_q{}'.format(q)]
        
        return result

    def get_parking_pc(self, row, row_prev=None, q=1):
        # 85-86
        # =
        #     ЕСЛИ(
        #           S$14=Итог!$M63;
        #           МИН(ОКРУГЛ(Итог!$L63*3*(1-Итог!$O63);0);Итог!$B63);
        #           ЕСЛИ(
        #                 S$14>Итог!$M63;
        #                 ЕСЛИ(
        #                       И(R63>0;S63=0);
        #                       Итог!$B63-СУММ($M86:R86);
        #                       МИН(Итог!$L63*3;Итог!$B63-СУММ($M86:R86))
        #                      );
        #                 0
        #                )
        #          )

        result = 0
        temp_build = round(8/40*(row['temp_build_q1'] + row['temp_build_q2'])/2)
        start_date_col = 'sale_start_period_q{}'.format(q)

        value_all = 214 if q==1 else 253

        if row['period'] == row[start_date_col]:
            result = round(temp_build*3*(1-0.3))
        elif row['period'] > row[start_date_col]:
            c_nonresidental_m2 = row['nonresidental_m2_q{}'.format(q)]
            past_nonresidental_m2 = row_prev['nonresidental_m2_q{}'.format(q)]

            sum_prev = row_prev['parking_pc_acc_q{}'.format(q)] if row_prev else 0
            end = value_all - sum_prev

            if c_nonresidental_m2 == 0 and past_nonresidental_m2 > 0:
                result = end
            else:
                result =min(end, temp_build*3)

            if result < 0:
                result = 0
  
        return result

    def get_parking_rubpc(self, row, row_prev=None, q=1):
        # 96-97
        parking_pc = row['parking_pc_q{}'.format(q)]
        result = 0
        if parking_pc:
            result = row['start_price_parking_q1'] if q==1 else row['start_price_parking_q2']
        return result

    def get_income_parking(self, row, row_prev=None, q=1):
        # 131-132
        parking_rubpc = row['parking_rubpc_q{}'.format(q)]
        parking_pc = row['parking_pc_q{}'.format(q)]

        result = parking_pc*parking_rubpc/1000
    
        return result

    def get_income_parking_acc(self, row, row_prev=None, q=1):
        result = row['income_parking_q{}'.format(q)]
        if row_prev is not None:
            result += row_prev['income_parking_acc_q{}'.format(q)]
        
        return result  

    def get_receipts_flats(self, row, row_prev=None, q=1):
        # 191-192
        part = 1
        result = 0
        if q == 1:
            offset = [0.27, 0.2, 0.53]
        else:
            offset = [0.27, 0.2, 0.53] #  [0.56, 0.07, 0.37]       

        construction_period = 12 if q==1 else 8

        if row['sale_start_period_q{}'.format(q)] + construction_period - 3 < row['period']:
         
            income_flats = row['income_flats_q{}'.format(q)]
            income_flats_past1 = row_prev['income_flats_q{}'.format(q)] if row_prev else 0
 
            result = income_flats_past1*part*1/3+income_flats*part*2/3

        elif row['sale_start_period_q{}'.format(q)] + construction_period - 3 == row['period']:                   
            income_flats = row['income_flats_q{}'.format(q)]            
            result = income_flats*part*2/3 
        else:            
                 
            income_flats = row['income_flats_q{}'.format(q)]
            income_flats_past1 = row_prev['income_flats_q{}'.format(q)] if row_prev else 0
            income_flats_past2 = row_prev['income_flats_q{}_past1'.format(q)] if row_prev else 0
            result = offset[0]*income_flats_past2 + offset[1]*income_flats_past1 + offset[2]*income_flats

        if row['sale_start_period_q{}'.format(q)] + construction_period -1 == row['period']:
            result += (row_prev['income_flats_acc_q{}'.format(q)] - row_prev['receipts_flats_acc_q{}'.format(q)] - 
                              row_prev['income_flats_q{}'.format(q)]/3)
            # Итог!$G40*СУММ($M109:AJ109)-СУММ($M191:AJ191)-AJ109*Итог!$G40*ЕСЛИ(Итог!$J$1=Итог!$S$4;1/3;0);0))

        return result

    def get_receipts_flats_acc(self, row, row_prev=None, q=1):
        result = row['receipts_flats_q{}'.format(q)]
        if row_prev is not None:
            result += row_prev['receipts_flats_acc_q{}'.format(q)]
        
        return result

    def get_receipts_nonresidental(self, row, row_prev=None, q=1):
        # 202-203
        part = 1
        result = 0
        if q == 1:
            offset = [0, 0, 1]
        else:
            offset = [0, 0, 1]       

        construction_period = 12 if q==1 else 8

        if row['sale_start_period_q{}'.format(q)] + construction_period - 3 < row['period']:
            income_nonresidental = row['income_nonresidental_q{}'.format(q)]
            income_nonresidental_past1 = row_prev['income_nonresidental_q{}'.format(q)] if row_prev else 0            
            result = income_nonresidental_past1*part*1/3+income_nonresidental*part*2/3

        elif row['sale_start_period_q{}'.format(q)] + construction_period - 3 == row['period']:
            income_nonresidental = row['income_nonresidental_q{}'.format(q)]    
            result = income_nonresidental*part*2/3 
        else:
            income_nonresidental = row['income_nonresidental_q{}'.format(q)]
            income_nonresidental_past1 = row_prev['income_nonresidental_q{}'.format(q)] if row_prev else 0  
            income_nonresidental_past2 =  row_prev['income_nonresidental_q{}_past1'.format(q)] if row_prev else 0       
            result = offset[0]*income_nonresidental_past2 + offset[1]*income_nonresidental_past1 + offset[2]*income_nonresidental

        if row['sale_start_period_q{}'.format(q)] + construction_period -1 == row['period']:
            result += (row_prev['income_nonresidental_acc_q{}'.format(q)] - row_prev['receipts_nonresidental_acc_q{}'.format(q)] - 
                              row_prev['income_nonresidental_q{}'.format(q)]/3)            

        return result 
    
    def get_receipts_nonresidental_acc(self, row, row_prev=None, q=1):
        result = row['receipts_nonresidental_q{}'.format(q)]
        if row_prev is not None:
            result += row_prev['receipts_nonresidental_acc_q{}'.format(q)]
        
        return result    
    
    def get_receipts_parking(self, row, row_prev=None, q=1):
        # 213-214
        part = 1
        result = 0

        if q == 1:
            offset = [0.27, 0.2, 0.53]
        else:
            offset = [0.27, 0.2, 0.53] # [0.56, 0.07, 0.37]       

        construction_period = 12 if q==1 else 8        

        if row['sale_start_period_q{}'.format(q)] + construction_period - 3 < row['period']:
            income_parking = row['income_parking_q{}'.format(q)]
            income_parking_past1 = row_prev['income_parking_q{}'.format(q)]  if row_prev else 0          
            result = income_parking_past1*part*1/3+income_parking*part*2/3

        elif row['sale_start_period_q{}'.format(q)] + construction_period - 3 == row['period']:         
            income_parking = row['income_parking_q{}'.format(q)]   
            result = income_parking*part*2/3 
        else:             
            income_parking = row['income_parking_q{}'.format(q)]
            income_parking_past1 = row_prev['income_parking_q{}'.format(q)]  if row_prev else 0    
            income_parking_past2 = row_prev['income_parking_q{}_past1'.format(q)]  if row_prev else 0          
            result = offset[0]*income_parking_past2 + offset[1]*income_parking_past1 + offset[2]*income_parking
        
        if row['sale_start_period_q{}'.format(q)] + construction_period -1 == row['period']:
            result += (row_prev['income_parking_acc_q{}'.format(q)] - row_prev['receipts_parking_acc_q{}'.format(q)] - 
                              row_prev['income_parking_q{}'.format(q)]/3)  
        return result 
    
    def get_receipts_parking_acc(self, row, row_prev=None, q=1):
        result = row['receipts_parking_q{}'.format(q)]
        if row_prev is not None:
            result += row_prev['receipts_parking_acc_q{}'.format(q)]
        
        return result   
    
    def get_revenue_after(self, row, row_prev=None, q=1):
        # 821-822
        construction_period = 12 if q==1 else 8
        result = 0
        if row['period'] >= row['sale_start_period_q{}'.format(q)] + construction_period:
            result = (row['receipts_flats_q{}'.format(q)] + 
                            row['receipts_nonresidental_q{}'.format(q)] +
                            row['receipts_parking_q{}'.format(q)])
        return result

    def get_escrow_cash_acc(self, row, row_prev=None, q=1):
        # 799-800
        construction_period = 12 if q==1 else 8
        result = 0
        
        if row['period'] <= row['sale_start_period_q{}'.format(q)] + construction_period - 1:
            result = row['receipts_flats_q{}'.format(q)] + row['receipts_nonresidental_q{}'.format(q)] + row['receipts_parking_q{}'.format(q)] 
            result += row_prev['escrow_cash_acc_q{}'.format(q)] if row_prev else 0

        return result

    def get_escrow_disclosure(self, row, row_prev=None, q=1):
        # 810-811
        construction_period = 12 if q==1 else 8
        result = 0

        if row['sale_start_period_q{}'.format(q)] + construction_period == row['period']:
            result = row_prev['escrow_cash_acc_q{}'.format(q)] if row_prev else 0

        return result

    def get_entrance_rights(self, row, row_prev=None, q=1):
        # 227-228
        result = 0
        if row['period'] == 9:
            all_value = -1200000000
            m2_q1 = 22087.8922
            m2_q2 = 57973.0078
            if q == 1:
                koef = m2_q1/(m2_q1+m2_q2)
            else:
                koef = m2_q2/(m2_q1+m2_q2)

            result = all_value*koef/1000

        return result

    def get_advertising(self, row, row_prev=None, q=1):
        # 250-251
        result = -(row['receipts_flats_q{}'.format(q)] + 
                   row['receipts_nonresidental_q{}'.format(q)] + 
                   row['receipts_parking_q{}'.format(q)])*0.035
        
        return result

    def get_commercial_staff(self, row, row_prev=None, q=1):
        # 261-262
        result = -(row['receipts_flats_q{}'.format(q)] + 
                   row['receipts_nonresidental_q{}'.format(q)] + 
                   row['receipts_parking_q{}'.format(q)])*0.008
        
        return result
    
    def get_remuneration_to_banks(self, row, row_prev=None, q=1):
        # 272-273
        result = -(row['receipts_flats_q{}'.format(q)] + 
                   row['receipts_nonresidental_q{}'.format(q)] + 
                   row['receipts_parking_q{}'.format(q)])*0.006
        
        return result    
    
    def get_adm_pers(self, row, row_prev=None, q=1):
        # 272-273
        result = -(row['receipts_flats_q{}'.format(q)] + 
                   row['receipts_nonresidental_q{}'.format(q)] + 
                   row['receipts_parking_q{}'.format(q)])*row['adm_percent_q{}'.format(q)]
        
        return result      

    def get_rent(self, row, row_prev=None, q=1):
        # 294-295
        return 0      
    
    def get_land_tax(self, row, row_prev=None, q=1):
        # 305-306
        if row['period'] <= 12:
            return 0

        if q==1:
            cost = 29315762.6709199 if row['period'] < row['sale_start_period_q{}'.format(q)] else 414055022.990907
        else:
            cost = 34345213.7390801 if row['period'] < row['sale_start_period_q{}'.format(q)] else 485090851.089093

        construction_period = 12 if q==1 else 8

        if row['period'] < row['sale_start_period_q{}'.format(q)]:
            stavka = 0.015
        elif (row['period'] - row['sale_start_period_q{}'.format(q)] >= 0 and 
                row['period'] - row['sale_start_period_q{}'.format(q)] < construction_period):
            stavka = 0.002
        elif row['period'] > row['sale_start_period_q{}'.format(q)] + construction_period - 1:
            stavka = 0
        else:
            stavka = 0.004

        result = -cost*stavka/4/1000
        
        return result  

    def get_taxes_from_gross_profit(self, row, row_prev=None):
        # 315
        if (row['escrow_disclosure_sum'] + row['revenue_after_sum']) == 0:
            result = 0
        else:
            result = -(
                row['receipts_flats_sum'] +
                row['receipts_nonresidental_sum'] +
                row['receipts_parking_sum'] +            
                
                row['advertising_sum'] +
                row['commercial_staff_sum'] +
                row['remuneration_to_banks_sum'] +
                row['adm_pers_sum'] +
                row['rent_sum'] +
                row['land_tax_sum'] +
                
                row['vat_nonresidental_sum'] +
                row['vat_parking_sum'] + 
                # ВРИ = 0
                # ВРИ% = 0
                # Соц обязательства = 0
                row['cost_price_sum'] + 
                # Акц займы = 0
                row['bridge_payout_sum'] + 
                row['escrow_payout_sum'] +
                row['entrance_rights_sum'])*0.25*(row['escrow_disclosure'] + row['revenue_after'])/(row['escrow_disclosure_sum'] + row['revenue_after_sum'])
        # if row['period'] == 24:
        #     print(row['receipts_flats_sum'] + row['receipts_nonresidental_sum'] + row['receipts_parking_sum'], 
        #           row['advertising_sum'],
        #           row['commercial_staff_sum'], 
        #           row['remuneration_to_banks_sum'],
        #           row['adm_pers_sum'],
        #           row['rent_sum'],
        #           row['land_tax_sum'],
        #           row['vat_nonresidental_sum'] + row['vat_parking_sum'],
        #           row['cost_price_sum'],
        #           row['bridge_payout_sum'],
        #           row['escrow_payout_sum'],
        #           row['entrance_rights_sum'], sep='\n'
        #           )
        if result > 0:
            result = 0

        return result

    def get_vat_nonresidental(self, row, row_pred=None):
        # 318
        # =-ЕСЛИ((СУММ(AI120:AI129*($K$2:$K$11<=AI14)*Итог!$G$51:$G$60)+Ч("Доход по нежилым после РВ")+
        #     СУММ(AI62:AI71*($K$2:$K$11<=AI14)*Итог!$G$51:$G$60*Итог!$AG$195:$AG$204)/1000+Ч("Расход по нежилым после РВ"))   /(1+Ставка_НДС)*Ставка_НДС<=0; 0;
        #     (СУММ(AI120:AI129*($K$2:$K$11<=AI14)*Итог!$G$51:$G$60)+Ч("Доход по нежилым после РВ")+
        #     СУММ(AI62:AI71*($K$2:$K$11<=AI14)*Итог!$G$51:$G$60*Итог!$AG$195:$AG$204)/1000+Ч("Расход по нежилым после РВ"))   /(1+Ставка_НДС)*Ставка_НДС)

        vat_koef = 0.2

        start_date_q1 = row['sale_start_period_q1']
        start_date_q2 = row['sale_start_period_q2']      
        end_date_q1 = start_date_q1 + 12 - 1
        end_date_q2 = start_date_q2 + 8 - 1       

        if row['period'] < end_date_q1:
            result_q1 = 0
        else:
            result_q1 = (row['income_nonresidental_q1'] + row['nonresidental_m2_q1']*(-64904.5772864956)/1000)*vat_koef/(1+vat_koef)
        
        if row['period'] < end_date_q2:
            result_q2 = 0
        else:
            result_q2 = (row['income_nonresidental_q2'] + row['nonresidental_m2_q2']*(-63484.6922195643)/1000)*vat_koef/(1+vat_koef)

        result = result_q1 + result_q2

        if result > 0:
            result = - result
        else:
            result = 0

        return result      

    def get_vat_parking(self, row, row_pred=None):
        # 319
        # =-ЕСЛИ((СУММ(AK131:AK140*($K$2:$K$11<=AK14)*Итог!$G$62:$G$71)+Ч("Доход м-местам после РВ")+
        #     СУММ(AK85:AK94*($K$2:$K$11<=AK14)*Итог!$C$62:$C$71*Итог!$G$62:$G$71*Итог!$AH$195:$AH$204)/1000+Ч("Расход по м-местам после РВ"))   /(1+Ставка_НДС)*Ставка_НДС<=0; 0;
        #     (СУММ(AK131:AK140*($K$2:$K$11<=AK14)*Итог!$G$62:$G$71)+Ч("Доход м-местам после РВ")+
        #     СУММ(AK85:AK94*($K$2:$K$11<=AK14)*Итог!$C$62:$C$71*Итог!$G$62:$G$71*Итог!$AH$195:$AH$204)/1000+Ч("Расход по м-местам после РВ"))   /(1+Ставка_НДС)*Ставка_НДС)

        vat_koef = 0.2

        start_date_q1 = row['sale_start_period_q1']
        start_date_q2 = row['sale_start_period_q2']      
        end_date_q1 = start_date_q1 + 12 - 1
        end_date_q2 = start_date_q2 + 8 - 1       

        if row['period'] < end_date_q1:
            result_q1 = 0
        else:
            result_q1 = (row['income_parking_q1'] + row['parking_pc_q1']*35*(-63500)/1000)*vat_koef/(1+vat_koef)
        
        if row['period'] < end_date_q2:
            result_q2 = 0
        else:
            result_q2 = (row['income_parking_q2'] + row['parking_pc_q2']*35*(-62500)/1000)*vat_koef/(1+vat_koef)

        result = result_q1 + result_q2

        if result > 0:
            result = - result
        else:
            result = 0

        return result            

    def get_smr_flats(self, row, row_pred=None, q=1):
        # 388-389
        construction_period = 12 if q==1 else 8
        if row['sale_start_period_q{}'.format(q)] + construction_period > row['period'] >= row['sale_start_period_q{}'.format(q)]:
            if q == 1:
                base = -22087.8922
            else:
                base = -57973.0078

            base = base - 2000

            result = (row['smr_flats_pc_q{}'.format(q)]*base)/construction_period
        else: 
            result = 0

        return result

    def get_finishing_wb(self, row, row_pred=None, q=1):
        # 410-411
        #     =ЕСЛИ(И(   AF2>1;   СЧЁТЕСЛИ($M2:AF2;">0")  >  ЕСЛИ(Итог!$J$1=Итог!$S$5;   4;   6));
        #   1/(СЧЁТЕСЛИ($M2:$ER2;">0")  -  ЕСЛИ(Итог!$J$1=Итог!$S$5;   4;   6));
        #   0)
        offset = 4
        construction_period = 12 if q==1 else 8
        if row['sale_start_period_q{}'.format(q)] + construction_period > row['period'] >= row['sale_start_period_q{}'.format(q)]+offset:
            if q == 1:
                base = -22087.8922
            else:
                base = 0
            base = base*22000

            result = base/(1000*(construction_period-offset))
        else: 
            result = 0

        return result

    def get_general_territory_preparation(self, row, row_pred=None, q=1):
        # 476-477
        if row['sale_start_period_q{}'.format(q)] - row['period'] in [1, 2]:  
            base = -1.16031722049293 if q==1 else -1.35938277950707000000

            base = base*10000000

            result = base/(1000*2)
        else:
            result = 0

        return result

    def get_dou(self, row, row_pred=None, q=1):
        # 498-499
        if row['period'] == 14:
            base = (-5547910.94888454*206/(22087.8922 + 57973.0078 + 4000)*(22087.8922 + 2000) if q == 1 else 
                        -5547910.94888454*206/(22087.8922 + 57973.0078 + 4000)*(57973.0078 + 2000))

            result = base/(1000)
        else:
            result = 0

        return result
    
    def get_schools(self, row, row_pred=None, q=1):
        # 520-521
        if row['period'] == 14:
            base = (-6351021.83536671*406/(22087.8922 + 57973.0078 + 4000)*(22087.8922 + 2000) if q == 1 else 
                        -6351021.83536671*406/(22087.8922 + 57973.0078 + 4000)*(57973.0078 + 2000))

            result = base/1000
        else:
            result = 0

        return result    
    
    def get_clinics(self, row, row_pred=None, q=1):
        # 542-543
        if row['period'] == 14:
            base = (-10035473.7963048*62/(22087.8922 + 57973.0078 + 4000)*(22087.8922 + 2000) if q == 1 else 
                        -10035473.7963048*62/(22087.8922 + 57973.0078 + 4000)*(57973.0078 + 2000))

            result = base/1000
        else:
            result = 0

        return result   
    
    def get_garages(self, row, row_pred=None, q=1):
        # 564-565
        construction_period = 12 if q==1 else 8
        if row['sale_start_period_q{}'.format(q)] + construction_period > row['period'] >= row['sale_start_period_q{}'.format(q)]:
            if q == 1:
                base = -214
            else:
                base = -253

            base = base*2100000

            result = base/(1000*construction_period)
        else: 
            result = 0

        return result   

    def get_offsite_networks(self, row, row_pred=None, q=1):
        # 586-587
        #=ЕСЛИ(И(AB2>0;СЧЁТЕСЛИ($M2:AB2;">0")<=СЧЁТЕСЛИ($M2:$ER2;">0")*2/3);1/(ОКРУГЛВНИЗ(СЧЁТЕСЛИ($M2:$ER2;">0")*2/3;0));0)
        construction_period = 12 if q==1 else 8
        if row['sale_start_period_q{}'.format(q)] + int(construction_period*2/3) > row['period'] >= row['sale_start_period_q{}'.format(q)]:
            # 22087.8922 + 57973.0078 + 4000

            base_all = 22087.8922 + 57973.0078 + 4000
            base = -(622701764/base_all)*(22087.8922 + 2000) if q==1 else -(622701764/base_all)*(57973.0078 + 2000)

            result = base/(1000*int(construction_period*2/3))                
        else: 
            result = 0

        return result      

    def get_onsite_networks(self, row, row_pred=None, q=1):
        # 608-609
        construction_period = 12 if q==1 else 8
        if row['sale_start_period_q{}'.format(q)] + construction_period > row['period'] >= row['sale_start_period_q{}'.format(q)]:
            base = -1.16031722049293 if q==1 else -1.35938277950707000000

            base = base*70000000

            result = base/(1000*construction_period)
        else: 
            result = 0

        return result
    
    def get_landscaping(self, row, row_pred=None, q=1):
        # 630-631
        construction_period = 12 if q==1 else 8
        duration = int(construction_period*2/3)

        offset = construction_period - duration

        if row['sale_start_period_q{}'.format(q)] + construction_period > row['period'] > row['sale_start_period_q{}'.format(q)] + offset-1:
            base = -1.16031722049293 if q==1 else -1.35938277950707000000

            base = base*100000000

            result = base/(1000*duration)
        else: 
            result = 0

        return result
    
    def get_aproval_of_design(self, row, row_pred=None, q=1):
        # 652-653
        duration = 4

        offset = 2

        if row['sale_start_period_q{}'.format(q)] + duration - offset > row['period'] > row['sale_start_period_q{}'.format(q)] - offset - 1:
            if q == 1:
                base = -3500
                mm = 22087.8922/0.72 + 2000/0.65 + 214*35 
                base = base*mm
            else:
                base = -2500
                mm = 57973.0078/0.72 + 2000/0.65 + 253*35 
                base = base*mm

            result = base/(1000*duration)
        else: 
            result = 0

        return result
    
    def get_unexpected_costs(self, row, row_pred=None, q=1):
        # 674-675
        # =СУММ(
        # AA454 - снос = 0
        # ;AA476 general_territory_preparation_q
        # ;AA388; smr_flats_q
        # AA608; onsite_networks_q
        # AA410; finishing_wb_q
        # AA432; Отделка чистовая = 0
        # AA498; dou_q
        # AA520; schools_q
        # AA542; clinics_q
        # AA564; garages_q
        # AA696; Запасная строка (себестоимость) = 0
        # AA630; landscaping_q
        # AA586; offsite_networks_q
        # AA652 aproval_of_design_q
        # )*Итог!$C338
        all_costs = (row['smr_flats_q{}'.format(q)] +
                    row['finishing_wb_q{}'.format(q)] +
                    row['general_territory_preparation_q{}'.format(q)] +
                    row['dou_q{}'.format(q)] +  
                    row['schools_q{}'.format(q)] +                       
                    row['clinics_q{}'.format(q)] +    
                    row['garages_q{}'.format(q)] +   
                    row['offsite_networks_q{}'.format(q)] +  
                    row['onsite_networks_q{}'.format(q)] +
                    row['landscaping_q{}'.format(q)] +                                                                                  
                    row['aproval_of_design_q{}'.format(q)])
        
        result = 0.01*all_costs

        return result

    def get_technical_customer(self, row, row_pred=None, q=1):
        # 685-686
        all_costs = (row['smr_flats_q{}'.format(q)] +
                    row['finishing_wb_q{}'.format(q)] +
                    row['general_territory_preparation_q{}'.format(q)] +
                    row['dou_q{}'.format(q)] + 
                    row['schools_q{}'.format(q)] +                        
                    row['clinics_q{}'.format(q)] +    
                    row['garages_q{}'.format(q)] +   
                    row['offsite_networks_q{}'.format(q)] +  
                    row['onsite_networks_q{}'.format(q)] +
                    row['landscaping_q{}'.format(q)] +                                                                                  
                    row['aproval_of_design_q{}'.format(q)])
        
        result = 0.01*all_costs 
        return result       

    def get_bridge_attracting(self, row, row_pred=None):
        # 719 -> # 769
        # V769=V750-V757
        if row['period'] >= row['sale_start_period_q1']:
            result = 0
        else:
            er = -row['entrance_rights']
            pr = -(row['advertising_q1'] + row['advertising_q2'] +
                        row['commercial_staff_q1'] + row['commercial_staff_q2'] + 
                        row['remuneration_to_banks_q1'] + row['remuneration_to_banks_q2'] + 
                        row['adm_pers_q1'] + row['adm_pers_q2'] + 
                        row['rent_q1'] +  row['rent_q2'] + 
                        row['land_tax_q1'] + row['land_tax_q2'] + row['cost_price']) 

            result = er + pr
        return result 
        
    def get_bridge_payout_md(self, row, row_pred=None):
        # 776
        return -row_pred['bridge_md_payout_sum_acc'] if row_pred and row['period'] == row['sale_start_period_q1'] else 0

    def get_bridge_payout_percent(self, row, row_pred=None):
        # 775
        return -row_pred['bridge_accural_percent'] if row_pred is not None else 0
            
    def get_bridge_md_payout_sum_acc(self, row, row_pred=None):
        # 777
        # =AA777+AB769+AB776+AA774+AB775
        result = row_pred['bridge_md_payout_sum_acc'] if row_pred is not None else 0
        
        # result += (row['bridge_payout_md'] + row['bridge_attracting'])
        result = result + ((row_pred['bridge_accural_percent'] if row_pred else 0) + row['bridge_payout_percent'] + row['bridge_payout_md'] + row['bridge_attracting'])
        return result
    
    def get_bridge_accural_percent(self, row, row_pred=None):
        # 774
        # =X772/4*X777
        percent = 0.07 + self.get_key_bid(row)

        return percent*row['bridge_md_payout_sum_acc']/4

    def get_bridge_attraction_for_current_expenses(self, row, row_pred=None, q=1):
        # 788-789
        construction_period = 12 if q ==1 else 8
        if row['period'] >= row['sale_start_period_q1'] and row['period'] <= row['sale_start_period_q1'] + construction_period-1:
            result = -(row['smr_flats_q{}'.format(q)] + 
                        row['finishing_wb_q{}'.format(q)] + 
                        row['general_territory_preparation_q{}'.format(q)] + 
                        row['dou_q{}'.format(q)] + 
                        row['schools_q{}'.format(q)] +                          
                        row['clinics_q{}'.format(q)] +  
                        row['garages_q{}'.format(q)] +  
                        row['offsite_networks_q{}'.format(q)] + 
                        row['onsite_networks_q{}'.format(q)] + 
                        row['landscaping_q{}'.format(q)] +                                                                                 
                        row['aproval_of_design_q{}'.format(q)] + 
                        row['unexpected_costs_q{}'.format(q)] + 
                        row['technical_customer_q{}'.format(q)] + 
                        row['advertising_q{}'.format(q)] +
                        row['commercial_staff_q{}'.format(q)] +
                        row['remuneration_to_banks_q{}'.format(q)] + 
                        row['adm_pers_q{}'.format(q)] + 
                        row['rent_q{}'.format(q)] +                  
                        row['land_tax_q{}'.format(q)])

        else:
            result = 0

        return result

    def get_escrow_attracting(self, row, row_pred=None):
        # 720
        construction_period = 12

        # payout_md = -row_pred['bridge_payout_md_sum_acc'] if self['period'] == row['sale_start_period_q1'] else 0
        payout_md = row['bridge_payout_md']
        attr_rs_1 = row['entrance_rights'] if (row['period'] >= row['sale_start_period_q1'] and row['period'] <= row['sale_start_period_q1'] + construction_period-1) else 0
        attr_c = row['bridge_attraction_for_current_expenses_q1'] + row['bridge_attraction_for_current_expenses_q2']
        result = -payout_md + attr_rs_1 + attr_c

        return result

    def get_key_bid(self, row):
        # 0 - 2021
        # 1:4 -2022 ...
        year = int(row['period']/4)+2022
        if year <= 2024:
            return row['key_bid_0']
        elif year == 2025:
            return row['key_bid_1']
        elif year == 2026:
            return row['key_bid_2']     
        elif year == 2027:
            return row['key_bid_3']  
        elif year >= 2028:
            return row['key_bid_4'] 
        else:
            return 0                       

    def get_bridge_payout(self, row, row_pred=None):
        # 780
        result_payout = row['bridge_payout_md'] + row['bridge_payout_percent']
        pf = row['bridge_payout_md'] if row['period'] == row['sale_start_period_q1'] else 0
        result = result_payout - pf
        return result 

    def get_escrow_attraction_transition_to_bridge(self, row, row_pred=None):
        # 785
        result = -row['bridge_payout_md']

        return result

    def get_escrow_attraction_entrance_after_q_1(self, row, row_pred=None):
        # 786
        construction_period = 12
        if row['period'] >= row['sale_start_period_q1'] and row['period'] < row['sale_start_period_q1'] + construction_period - 1:
            result = -row['entrance_rights']
        else:
            result = 0

        return result
    
    def get_escrow_attraction_entrance_all(self, row, row_pred=None):
        # 784
        result = (row['escrow_attraction_transition_to_bridge'] + row['escrow_attraction_entrance_after_q_1'] 
                + row['escrow_current_expenses_q1'] + row['escrow_current_expenses_q2'])

        return result

    def get_escrow_current_expenses(self, row, row_pred=None, q=1):
        # 788-789
        construction_period = 12 if q ==1 else 8
        if row['period'] >= row['sale_start_period_q{}'.format(q)] and row['period'] < row['sale_start_period_q{}'.format(q)] + construction_period:
            result = -(row['smr_flats_q{}'.format(q)] + 
                        row['finishing_wb_q{}'.format(q)] + 
                        row['general_territory_preparation_q{}'.format(q)] + 
                        row['dou_q{}'.format(q)] + 
                        row['schools_q{}'.format(q)] +                          
                        row['clinics_q{}'.format(q)] +  
                        row['garages_q{}'.format(q)] +  
                        row['offsite_networks_q{}'.format(q)] + 
                        row['onsite_networks_q{}'.format(q)] + 
                        row['landscaping_q{}'.format(q)] +                                                                                 
                        row['aproval_of_design_q{}'.format(q)] + 
                        row['unexpected_costs_q{}'.format(q)] + 
                        row['technical_customer_q{}'.format(q)] + 
                        row['advertising_q{}'.format(q)] +
                        row['commercial_staff_q{}'.format(q)] +
                        row['remuneration_to_banks_q{}'.format(q)] + 
                        row['adm_pers_q{}'.format(q)] + 
                        row['rent_q{}'.format(q)] +                  
                        row['land_tax_q{}'.format(q)])
        else:
            result = 0

        return result
    
    def get_escrow_current_expenses_flow(self, row, row_pred=None, q=1):
        # 843-844
        construction_period = 12 if q ==1 else 8
        if row['period'] > row['sale_start_period_q{}'.format(q)] + construction_period-1:
            result = (row['smr_flats_q{}'.format(q)] + 
                        row['finishing_wb_q{}'.format(q)] + 
                        row['general_territory_preparation_q{}'.format(q)] + 
                        row['dou_q{}'.format(q)] + 
                        row['schools_q{}'.format(q)] +                          
                        row['clinics_q{}'.format(q)] +  
                        row['garages_q{}'.format(q)] +  
                        row['offsite_networks_q{}'.format(q)] + 
                        row['onsite_networks_q{}'.format(q)] + 
                        row['landscaping_q{}'.format(q)] +                                                                                 
                        row['aproval_of_design_q{}'.format(q)] + 
                        row['unexpected_costs_q{}'.format(q)] + 
                        row['technical_customer_q{}'.format(q)] + 
                        row['advertising_q{}'.format(q)] +
                        row['commercial_staff_q{}'.format(q)] +
                        row['remuneration_to_banks_q{}'.format(q)] + 
                        row['adm_pers_q{}'.format(q)] + 
                        row['rent_q{}'.format(q)] +                  
                        row['land_tax_q{}'.format(q)])
        else:
            result = 0

        return result    

    def get_escrow_flow(self, row, row_pred=None, q=1):
        # 854-855
        # =AN832+AN843
        escrow_disclosure_revenue_after = row['escrow_disclosure_q{}'.format(q)] + row['revenue_after_q{}'.format(q)]

        escrow_current_expenses_flow = row['escrow_current_expenses_flow_q{}'.format(q)]

        result = escrow_current_expenses_flow + escrow_disclosure_revenue_after

        return result

    def get_escrow_debt_rest(self, row, row_pred=None, q=None):
        # 866-867
        # AE866 =AD866+-AE788+AD918-AD929-AE785-AE786+Ч("772 - бридж, 773 - вход после бриджа")
        construction_period = 12
        if q:
            prev_escrow_debt_rest = row_pred['escrow_debt_rest_q{}'.format(q)] if row_pred is not None else 0 # 866-867
            prev_escrow_accural_percent = row_pred['escrow_accural_percent_q{}'.format(q)] if row_pred is not None else 0 # 918-919
            prev_escrow_repayment = row_pred['escrow_repayment_q{}'.format(q)] if row_pred is not None else 0 # 929-930
            c_attraction_for_current_expenses = row['bridge_attraction_for_current_expenses_q{}'.format(q)] # 788-789

            result = (prev_escrow_debt_rest - c_attraction_for_current_expenses + 
                    prev_escrow_accural_percent - prev_escrow_repayment)
            
            # if row['period'] in (25, 26) and q==2:
            #     print(row['period'], prev_escrow_debt_rest, c_attraction_for_current_expenses, prev_escrow_accural_percent, prev_escrow_repayment, result)

            if q == 1:
                attracting_to_bridge = -row['bridge_payout_md']
                attr_rs_1 = row['entrance_rights'] if (row['period'] >= row['sale_start_period_q1'] and row['period'] <= row['sale_start_period_q1'] + 
                                                       construction_period-1) else 0                 
                result -= (attracting_to_bridge + attr_rs_1)

        else:
            # 865
            # =AB865+-AC784+AB917-AB928
            prev_escrow_debt_rest = row_pred['escrow_debt_rest'] if row_pred is not None else 0 # 865
            prev_escrow_accural_percent = row_pred['escrow_accural_percent'] if row_pred is not None else 0 # 917
            prev_escrow_repayment = row_pred['escrow_repayment'] if row_pred is not None else 0 # 928
            c_escrow_attraction_entrance_all = row['escrow_attraction_entrance_all'] # 784 = 785+786+787!!!!!!!!!!!          
       
            result = (prev_escrow_debt_rest - c_escrow_attraction_entrance_all + 
                    prev_escrow_accural_percent - prev_escrow_repayment)
            # if row['period'] == 25:
            #     print(prev_escrow_debt_rest, c_escrow_attraction_entrance_all, prev_escrow_accural_percent, prev_escrow_repayment, result)
            
            # if row['period'] in (16, 17):
            #     print(row['period'], prev_escrow_debt_rest, c_escrow_attraction_entrance_all, prev_escrow_accural_percent, prev_escrow_repayment, result)              
                              
        return result
    
    def get_escrow_cash_rest(self, row, row_pred=None, q=1):
        # 798
        # 799-800
        # AG799=ЕСЛИ(AG$14<=$K2;   СУММ(AG191;AG202;AG213)+AF799;   0)
        construction_period = 12 if q ==1 else 8
        if row['period'] <= (row['sale_start_period_q{}'.format(q)] + construction_period - 1):
            c_pred = row_pred['escrow_cash_rest_q{}'.format(q)] if row_pred is not None else 0
            result = c_pred + row['receipts_flats_q{}'.format(q)] + row['receipts_nonresidental_q{}'.format(q)] + row['receipts_parking_q{}'.format(q)]
        else:
            result = 0

        return result

    def get_escrow_cover_koef(self, row, row_pred=None, q=None):
        # 877
        # 888-889
        # =ЕСЛИОШИБКА(-AE798/AE865;0)
        # if q:             
        #     if row['escrow_debt_rest_q{}'.format(q)] != 0:
        #         result = -row['escrow_cash_rest_q{}'.format(q)]/row['escrow_debt_rest_q{}'.format(q)]
        #     else:
        #         result = 0
        # else:
        # if row['period'] == 16:
        #     print(row['escrow_cash_rest'], row['escrow_debt_rest'])   

        if row['escrow_debt_rest'] != 0:
            result = -row['escrow_cash_rest']/row['escrow_debt_rest']
        else:
            result = 0            

        return result
    
    def get_escrow_bid(self, row, row_pred=None, q=None):
        # 877-878
        # =ЕСЛИ(AG866=0;0;
        # ЕСЛИ(AG888<$I$911;МАКС(AG$907*(1-AG888)+AG$908*AG888;AG$908);
        # ЕСЛИ(AG888<$I$912;AG$912;
        # ЕСЛИ(AG888<$I$913;AG$913;
        # ЕСЛИ(AG888<$I$914;AG$914;
        # ЕСЛИ(AG888<$I$915;AG$915;AG$916))))))
        # 1,10
        # 1,20
        # 1,30
        # 1,40
        # 1,50
        base_bid = self.get_key_bid(row) + 0.045
        special_bid = 0.035

        # if row['period'] == 16 and q==2:
        #     print(row['escrow_cover_koef_q{}'.format(q)], base_bid, special_bid)
        if q:
            if row['escrow_debt_rest_q{}'.format(q)] == 0:
                result = 0
            elif row['escrow_cover_koef_q{}'.format(q)] < 1.1:
                result = max(base_bid*(1-row['escrow_cover_koef_q{}'.format(q)]) +special_bid*row['escrow_cover_koef_q{}'.format(q)], special_bid)
            elif row['escrow_cover_koef_q{}'.format(q)] < 1.2:
                result = 0.0229
            elif row['escrow_cover_koef_q{}'.format(q)] < 1.3:    
                result = 0.0159
            elif row['escrow_cover_koef_q{}'.format(q)] < 1.4:    
                result = 0.0088
            elif row['escrow_cover_koef_q{}'.format(q)] < 1.5:    
                result = 0.0018       
            else:
                result = 0.0001        
        else:
            if row['escrow_debt_rest'] == 0:
                result = 0
            elif row['escrow_cover_koef'] < 1.1:
                result = max(base_bid*(1-row['escrow_cover_koef']) +special_bid*row['escrow_cover_koef'], special_bid)
            elif row['escrow_cover_koef'] < 1.2:
                result = 0.0229
            elif row['escrow_cover_koef'] < 1.3:    
                result = 0.0159
            elif row['escrow_cover_koef'] < 1.4:    
                result = 0.0088
            elif row['escrow_cover_koef'] < 1.5:    
                result = 0.0018       
            else:
                result = 0.0001   


        return result

    def get_escrow_accural_percent(self, row, row_pred=None, q=None):

        if q:
            # 918-919
            # =(AH866-AH929)*AH877/4
            result = (row['escrow_debt_rest_q{}'.format(q)] - row['escrow_repayment_q{}'.format(q)])*row['escrow_bid_q{}'.format(q)]/4

            # if row['period'] == 24 and q==2:
            #     print(row['escrow_debt_rest_q{}'.format(q)], row['escrow_repayment_q{}'.format(q)], row['escrow_bid_q{}'.format(q)])            
        else:
            # 917
            # =(AD865-AD928)*AD876/4\
            # if row['period'] == 24:
            #     print(q, row['escrow_debt_rest'], row['escrow_repayment'], row['escrow_bid'])             
            result = (row['escrow_debt_rest'] - row['escrow_repayment'])*row['escrow_bid']/4

        return result    
    
    def get_escrow_flow_with_repayment(self, row, row_pred=None):
        # 864
        # AJ$864 =AJ853+AI864+AI928
        c_predv = row_pred['escrow_flow_with_repayment'] if row_pred is not None else 0
        escrow_repayment_predv = row_pred['escrow_repayment_q1'] + row_pred['escrow_repayment_q2'] if row_pred is not None else 0

        result = row['escrow_flow_q1'] + row['escrow_flow_q2'] + c_predv + escrow_repayment_predv

        return result

    def get_escrow_repayment(self, row, row_pred=None, q=None):
        if q:
            # 929-930
            # =-МИН(-AJ867; AJ$864+Ч("Совокупный поток для обслуживания кредита")+СУММ(AJ$929:AJ929))
            result_sum_pred = 0
            for qq in range(1, q):
                result_sum_pred += row['escrow_repayment_q{}'.format(qq)] # 929-930

            result = -min(-row['escrow_debt_rest_q{}'.format(q)], row['escrow_flow_with_repayment'] + result_sum_pred)
        else:
            # 928
            result = -min(-row['escrow_debt_rest'], row['escrow_flow_with_repayment'])

        return result

    def get_escrow_repayment_percent(self, row, row_pred=None, q=1, escrow_repayment_c=0, escrow_repayment_percent_c=0):
        # 940-941
        # =ЕСЛИ(СЧЁТЕСЛИ($M930:AN930;"<>"&0)=0+Ч("Погашения ещё не было?");
        #     0;
        #     ЕСЛИ(СЧЁТЕСЛИ($M941:AM941;"<>"&0)=0+Ч("Выплаты % ещё не было?");
        #         МАКС(СУММ($M919:AM919);AN930);
        #         ЕСЛИ(AN930=0+Ч("В текущем периоде погашения нет?");
        #             0;
        #             МАКС(СУММ($M919:AM919)-СУММ($M941:AM941);AN930))))          

        is_repayment = row['escrow_repayment_q{}'.format(q)] != 0
        was_repayment = escrow_repayment_c != 0 or is_repayment # 930
        was_payment = escrow_repayment_percent_c != 0 # 940        

        # if row['period'] == 24:
        #     print(was_repayment, was_payment, is_repayment)

        if not was_repayment:
            result = 0
        elif was_repayment and not was_payment:
            result = max(row_pred['escrow_accural_percent_acc_q{}'.format(q)], row['escrow_repayment_q{}'.format(q)])
        elif was_repayment and was_payment and not is_repayment:
            result = 0
        else:
            result = max(row_pred['escrow_accural_percent_acc_q{}'.format(q)] - row_pred['escrow_repayment_percent_acc_q{}'.format(q)], 
                         row['escrow_repayment_q{}'.format(q)])

        return result

    def get_escrow_accural_percent_acc(self, row, row_pred=None, q=1):
        return (row_pred['escrow_accural_percent_acc_q{}'.format(q)] if row_pred else 0) + row['escrow_accural_percent_q{}'.format(q)]
    
    def get_escrow_repayment_percent_acc(self, row, row_pred=None, q=1):
        return (row_pred['escrow_repayment_percent_acc_q{}'.format(q)] if row_pred else 0) + row['escrow_repayment_percent_q{}'.format(q)]    

    def get_escrow_payout(self, row, row_pred=None):
        # 723
        return row['escrow_repayment_percent_q1'] + row['escrow_repayment_percent_q2']
   
    def get_bridge_refund(self, row, row_pred=None):
        # 725
        # AB725=ЕСЛИ($I$775="Выплата";AB779;ЕСЛИ(AB779<>0;-$K$769;0))
        # AB779=ЕСЛИ($I$775="Выплата";AB776;(AB776+AB775))
        return self.get_bridge_payout_md(row, row_pred)

    def get_escrow_md(self, row, row_pred=None):
        # 950
        # =МИН(AK928-AK939;0)
        return min(row['escrow_repayment'] - row['escrow_repayment_percent'], 0) 

    def get_escrow_refund(self, row, row_pred=None):
        # 726
        # =AJ950
        return row['escrow_md']
    
    def get_project_expenses(self, row, row_pred=None):
        result = row['entrance_rights'] + row['realization_costs'] + row['cost_price'] + row['financial_activities']
        return result

    def get_financial_activities(self, row, row_pred=None):
        # 717
        result = (row['bridge_attracting'] + 
                  row['escrow_attracting'] + 
                  row['bridge_payout'] + 
                  row['escrow_payout'] + 
                  row['bridge_refund'] + 
                  row['escrow_refund'])
        
        return result

    def get_net_profit(self, row, row_pred=None):
        result = row['project_expences'] + row['escrow_disclosure'] + row['revenue_after']
        return result        

    def calculate_ds(self, dataset, field='', stage=100):
        dataset = dataset.copy()

        self.initialize_fields(dataset)

        if field:
            if field in ['flats_pc_q1', 'flats_pc_q2']:
                stage = 1
            elif field in ['income_flats_q1', 'income_flats_q2']:
                stage = 2
            elif field in ['income_nonresidental_q1', 'income_nonresidental_q2', 'income_parking_q1', 'income_parking_q2']:
                stage = 3    
            elif field in ['receipts_flats_q1', 'receipts_flats_q2', 'receipts_nonresidental_q1', 'receipts_nonresidental_q2', 'receipts_parking_q1', 'receipts_parking_q2']:
                stage = 4 
            elif field in ['revenue_after_q1', 'revenue_after_q2', 'revenue_after']: 
                stage = 5       
            elif field in ['disclosure_escrow_q1', 'disclosure_escrow_q2', 'disclosure_escrow']: 
                stage = 6 
            elif field in ['entrance_rights_q1', 'entrance_rights_q2', 'entrance_rights']: 
                stage = 7    
                                        
            else:
                stage = 100

        row_prev = None

        dataset_list = dataset.to_dict(orient='records')
        flats_pc_q1_all = 0
        flats_pc_q2_all = 0

        for ind, row in enumerate(dataset_list):

            for q in [2, 1]:
                res_q = self.get_flats_pc(row, row_prev, q=q) 
                row['flats_pc_q{}'.format(q)] = res_q 
                if row['flats_pc_q{}'.format(q)] > 0:
                    if q==1:
                        flats_pc_q1_all += 1
                    else:
                        flats_pc_q2_all += 1

                row['flats_pc_q{}_acc'.format(q)] = self.get_flats_pc_acc(row, row_prev, q=q)
                if res_q:
                    row['flats_pc_q{}_all_qty'.format(q)] += 1

            if stage > 1:
                for q in [2, 1]:
                    row['flats_m2_q{}'.format(q)] = self.get_flats_m2(row, row_prev, q)
                    row['flats_rubm2_q{}'.format(q)] = self.get_flats_rubm2(row, row_prev, q)

                    row['income_flats_q{}'.format(q)] = self.get_income_flats(row, row_prev, q)

                    row['income_flats_q{}_past1'.format(q)] = row_prev['income_flats_q{}'.format(q)] if row_prev else 0
                    row['income_flats_acc_q{}'.format(q)] = self.get_income_flats_acc(row, row_prev, q)

            row_prev = row

        for ind, row in enumerate(dataset_list):
            for q in [2, 1]:
                row['flats_pc_q{}_all_qty'.format(q)] = flats_pc_q1_all if q == 1 else flats_pc_q2_all

        if stage > 2:        
            row_prev = None
            for ind, row in enumerate(dataset_list):
                for q in [2, 1]:
                    row['nonresidental_m2_q{}'.format(q)] = self.get_nonresidental_m2(row, row_prev, q)
                    row['nonresidental_rubm2_q{}'.format(q)] = self.get_nonresidental_rubm2(row, row_prev, q)
                    row['income_nonresidental_q{}'.format(q)] = self.get_income_nonresidental(row, row_prev, q)
                    row['income_nonresidental_q{}_past1'.format(q)] = row_prev['income_nonresidental_q{}'.format(q)] if row_prev else 0   
                    row['income_nonresidental_acc_q{}'.format(q)] = self.get_income_nonresidental_acc(row, row_prev, q)                                 

                    row['parking_pc_q{}'.format(q)] = self.get_parking_pc(row, row_prev, q)
                    row['parking_pc_acc_q{}'.format(q)] = self.get_parking_pc_acc(row, row_prev, q)                    
                    row['parking_rubpc_q{}'.format(q)] = self.get_parking_rubpc(row, row_prev, q)
                    row['income_parking_q{}'.format(q)] = self.get_income_parking(row, row_prev, q) 
                    row['income_parking_q{}_past1'.format(q)] = row_prev['income_parking_q{}'.format(q)]  if row_prev else 0                      
                    row['income_parking_acc_q{}'.format(q)] = self.get_income_parking_acc(row, row_prev, q) 

                    if stage > 3:
                        row['receipts_flats_q{}'.format(q)] = self.get_receipts_flats(row, row_prev, q)
                        row['receipts_flats_acc_q{}'.format(q)] = self.get_receipts_flats_acc(row, row_prev, q)
                        row['receipts_nonresidental_q{}'.format(q)] = self.get_receipts_nonresidental(row, row_prev, q) 
                        row['receipts_nonresidental_acc_q{}'.format(q)] = self.get_receipts_nonresidental_acc(row, row_prev, q)                        
                        row['receipts_parking_q{}'.format(q)] = self.get_receipts_parking(row, row_prev, q) 
                        row['receipts_parking_acc_q{}'.format(q)] = self.get_receipts_parking_acc(row, row_prev, q) 

                        if stage > 4:
                            row['revenue_after_q{}'.format(q)] = self.get_revenue_after(row, row_prev, q)
                if stage > 4:
                    row['revenue_after'] = row['revenue_after_q1'] + row['revenue_after_q2'] 
                
                    # for q in [2, 1]:
                    #     row['entrance_rights_q{}'.format(q)] = self.get_entrance_rights(row, row_prev, q)

                    # row['entrance_rights'] = row['entrance_rights_q1'] + row['entrance_rights_q2']

                row_prev = row            
    
        if stage > 5:
            row_prev = None
            for ind, row in enumerate(dataset_list):
                for q in [2, 1]:
                    row['escrow_cash_acc_q{}'.format(q)] = self.get_escrow_cash_acc(row, row_prev, q)
                    row['escrow_disclosure_q{}'.format(q)] = self.get_escrow_disclosure(row, row_prev, q) 

                row['escrow_disclosure'] = row['escrow_disclosure_q1'] + row['escrow_disclosure_q2']
                row_prev = row

        if stage > 6:
            row_prev = None
            for ind, row in enumerate(dataset_list):
                for q in [2, 1]:
                    row['entrance_rights_q{}'.format(q)] = self.get_entrance_rights(row, row_prev, q)
                row['entrance_rights'] = row['entrance_rights_q1'] + row['entrance_rights_q2']                

                row_prev = row 

        if stage > 7:
            row_prev = None

            for ind, row in enumerate(dataset_list):
                for q in [2, 1]:
                    row['advertising_q{}'.format(q)] = self.get_advertising(row, row_prev, q)
                    row['commercial_staff_q{}'.format(q)] = self.get_commercial_staff(row, row_prev, q)
                    row['remuneration_to_banks_q{}'.format(q)] = self.get_remuneration_to_banks(row, row_prev, q)  
                    row['adm_pers_q{}'.format(q)] = self.get_adm_pers(row, row_prev, q) 
                    row['rent_q{}'.format(q)] = self.get_rent(row, row_prev, q)                  
                    row['land_tax_q{}'.format(q)] = self.get_land_tax(row, row_prev, q)

                row['vat_nonresidental'] = self.get_vat_nonresidental(row, row_prev)
                row['vat_parking'] = self.get_vat_parking(row, row_prev)

                # Изменение ВРИ (Собственность) = 0
                # Изменение ВРИ (Собственность) (%) = 0
                # Соц. обязательства = 0
                # Доп строка = 0

                row['realization_costs'] = (row['advertising_q1'] + row['advertising_q2'] +
                                        row['commercial_staff_q1'] + row['commercial_staff_q2'] + 
                                        row['remuneration_to_banks_q1'] + row['remuneration_to_banks_q2'] + 
                                        row['adm_pers_q1'] + row['adm_pers_q2'] + 
                                        row['rent_q1'] +  row['rent_q2'] + 
                                        row['land_tax_q1'] + row['land_tax_q2'] + 
                                        row['vat_nonresidental'] +
                                        row['vat_parking']
                                        ) # Еще потом добавить taxes_from_gross_profit             

                row_prev = row      

        if stage > 8:
            for ind, row in enumerate(dataset_list):
                for q in [2, 1]:
                    row['smr_flats_q{}'.format(q)] = self.get_smr_flats(row, row_prev, q)
                    row['finishing_wb_q{}'.format(q)] = self.get_finishing_wb(row, row_prev, q)
                    # Отделка чистовая = 0
                    # Снос = 0
                    row['general_territory_preparation_q{}'.format(q)] = self.get_general_territory_preparation(row, row_prev, q)
                    row['dou_q{}'.format(q)] = self.get_dou(row, row_prev, q) 
                    row['schools_q{}'.format(q)] = self.get_schools(row, row_prev, q)                        
                    row['clinics_q{}'.format(q)] = self.get_clinics(row, row_prev, q)   
                    row['garages_q{}'.format(q)] = self.get_garages(row, row_prev, q)   
                    row['offsite_networks_q{}'.format(q)] = self.get_offsite_networks(row, row_prev, q)  
                    row['onsite_networks_q{}'.format(q)] = self.get_onsite_networks(row, row_prev, q)
                    row['landscaping_q{}'.format(q)] = self.get_landscaping(row, row_prev, q)                                                                                  
                    row['aproval_of_design_q{}'.format(q)] = self.get_aproval_of_design(row, row_prev, q)
                    row['unexpected_costs_q{}'.format(q)] = self.get_unexpected_costs(row, row_prev, q)
                    row['technical_customer_q{}'.format(q)] = self.get_technical_customer(row, row_prev, q)

                row['cost_price'] = (row['smr_flats_q1'] + row['smr_flats_q2'] +
                                    row['finishing_wb_q1'] + row['finishing_wb_q2'] +
                                    row['general_territory_preparation_q1'] + row['general_territory_preparation_q2'] +
                                    row['dou_q1'] + row['dou_q2'] +  
                                    row['schools_q1'] + row['schools_q2'] +                                      
                                    row['clinics_q1'] + row['clinics_q2'] +  
                                    row['garages_q1'] + row['garages_q2'] +  
                                    row['offsite_networks_q1'] + row['offsite_networks_q2'] + 
                                    row['onsite_networks_q1'] + row['onsite_networks_q2'] + 
                                    row['landscaping_q1'] + row['landscaping_q2'] +                                                                                 
                                    row['aproval_of_design_q1'] + row['aproval_of_design_q2'] + 
                                    row['unexpected_costs_q1'] + row['unexpected_costs_q2'] +
                                    row['technical_customer_q1'] + row['technical_customer_q2'])              
                    
        if stage > 9:
            row_prev = None
            escrow_repayment_c = [0, 0]
            escrow_repayment_percent_c = [0, 0]
            for ind, row in enumerate(dataset_list):
                # Привелечение Акционерные займы = 0
                row['bridge_attracting'] = self.get_bridge_attracting(row, row_prev)

                row['bridge_payout_md'] = self.get_bridge_payout_md(row, row_prev)
                row['bridge_payout_percent'] = self.get_bridge_payout_percent(row, row_prev)
                row['bridge_md_payout_sum_acc'] = self.get_bridge_md_payout_sum_acc(row, row_prev)
                row['bridge_accural_percent'] = self.get_bridge_accural_percent(row, row_prev)
                row['bridge_attraction_for_current_expenses_q1'] = self.get_bridge_attraction_for_current_expenses(row, row_prev, q=1)
                row['bridge_attraction_for_current_expenses_q2'] = self.get_bridge_attraction_for_current_expenses(row, row_prev, q=2)
                row['bridge_attraction_for_current_expenses'] = row['bridge_attraction_for_current_expenses_q1'] + row['bridge_attraction_for_current_expenses_q2']
                row['escrow_attracting'] = self.get_escrow_attracting(row, row_prev)
                # Выплата процентов по акционерным займам = 0

                row['bridge_payout'] = self.get_bridge_payout(row, row_prev)

                for q in [2, 1]:   
                    row['escrow_current_expenses_q{}'.format(q)] = self.get_escrow_current_expenses(row, row_prev, q)
                    row['escrow_current_expenses_flow_q{}'.format(q)] = self.get_escrow_current_expenses_flow(row, row_prev, q)                    
                    row['escrow_flow_q{}'.format(q)] = self.get_escrow_flow(row, row_prev, q) 
                    row['escrow_debt_rest_q{}'.format(q)] = self.get_escrow_debt_rest(row, row_prev, q)
                
                row['escrow_attraction_transition_to_bridge'] = self.get_escrow_attraction_transition_to_bridge(row, row_prev)                  
                row['escrow_attraction_entrance_after_q_1'] = self.get_escrow_attraction_entrance_after_q_1(row, row_prev)                
                row['escrow_attraction_entrance_all'] = self.get_escrow_attraction_entrance_all(row, row_prev) 

                row['escrow_debt_rest'] = self.get_escrow_debt_rest(row, row_prev)
                for q in [2, 1]:   
                    row['escrow_cash_rest_q{}'.format(q)] = self.get_escrow_cash_rest(row, row_prev, q)
                row['escrow_cash_rest'] = row['escrow_cash_rest_q1'] + row['escrow_cash_rest_q2']

                row['escrow_current_expenses'] = row['escrow_current_expenses_q1'] +  row['escrow_current_expenses_q2']
                row['escrow_current_expenses_flow'] = row['escrow_current_expenses_flow_q1'] +  row['escrow_current_expenses_flow_q2']
                row['escrow_flow_with_repayment'] = self.get_escrow_flow_with_repayment(row, row_prev) 
                
                for q in [1, 2]: 
                    row['escrow_cover_koef_q{}'.format(q)] = self.get_escrow_cover_koef(row, row_prev) 
                    row['escrow_bid_q{}'.format(q)] = self.get_escrow_bid(row, row_prev, q)   
                    row['escrow_repayment_q{}'.format(q)] = self.get_escrow_repayment(row, row_prev, q)                                      
                    row['escrow_accural_percent_q{}'.format(q)] = self.get_escrow_accural_percent(row, row_prev, q)
                    row['escrow_accural_percent_acc_q{}'.format(q)] = self.get_escrow_accural_percent_acc(row, row_prev, q)                    
                    row['escrow_repayment_percent_q{}'.format(q)] = self.get_escrow_repayment_percent(row, row_prev, q, escrow_repayment_c[q-1], escrow_repayment_percent_c[q-1])
                    row['escrow_repayment_percent_acc_q{}'.format(q)] = self.get_escrow_repayment_percent_acc(row, row_prev, q)                     
                    if row['escrow_repayment_q{}'.format(q)] != 0:
                        escrow_repayment_c[q-1] += 1
                    if row['escrow_repayment_percent_q{}'.format(q)] != 0:
                        escrow_repayment_percent_c[q-1] += 1

                row['escrow_cover_koef'] = self.get_escrow_cover_koef(row, row_prev)                 
                row['escrow_bid'] = self.get_escrow_bid(row, row_prev)  
                
                row['escrow_repayment'] = self.get_escrow_repayment(row, row_prev)  
                row['escrow_accural_percent'] = self.get_escrow_accural_percent(row, row_prev) 
                row['escrow_repayment_percent'] = row['escrow_repayment_percent_q1'] + row['escrow_repayment_percent_q2'] 
                row['escrow_payout'] = self.get_escrow_payout(row, row_prev)

                row_prev = row              

            for ind, row in enumerate(dataset_list):                                                                                        
                # Возврат акционерных займов = 0

                row['bridge_refund'] = self.get_bridge_refund(row, row_prev)
                row['escrow_md'] = self.get_escrow_md(row, row_prev)                
                row['escrow_refund'] = self.get_escrow_refund(row, row_prev)

                row['financial_activities'] = self.get_financial_activities(row, row_prev)

                row_prev = row
                   
        if stage > 10:

            receipts_flats_sum = 0
            receipts_nonresidental_sum = 0
            receipts_parking_sum = 0    

            advertising_sum = 0
            commercial_staff_sum = 0
            remuneration_to_banks_sum = 0
            adm_pers_sum = 0
            rent_sum = 0
            land_tax_sum = 0
            vat_nonresidental_sum = 0
            vat_parking_sum = 0
            cost_price_sum = 0
            bridge_payout_sum = 0
            escrow_payout_sum = 0
            entrance_rights_sum = 0
            escrow_disclosure_sum = 0
            revenue_after_sum = 0

            for ind, row in enumerate(dataset_list):
                for q in [1, 2]:

                    receipts_flats_sum += row['receipts_flats_q{}'.format(q)]
                    receipts_nonresidental_sum += row['receipts_nonresidental_q{}'.format(q)]
                    receipts_parking_sum += row['receipts_parking_q{}'.format(q)]   

                    advertising_sum += row['advertising_q{}'.format(q)]
                    commercial_staff_sum += row['commercial_staff_q{}'.format(q)]
                    remuneration_to_banks_sum += row['remuneration_to_banks_q{}'.format(q)]
                    adm_pers_sum += row['adm_pers_q{}'.format(q)]
                    rent_sum += row['rent_q{}'.format(q)]
                    land_tax_sum += row['land_tax_q{}'.format(q)]

                vat_nonresidental_sum += row['vat_nonresidental']
                vat_parking_sum += row['vat_parking']  
                
                cost_price_sum += row['cost_price']
                bridge_payout_sum += row['bridge_payout']
                escrow_payout_sum += row['escrow_payout']    
                entrance_rights_sum += row['entrance_rights']  
                escrow_disclosure_sum += row['escrow_disclosure']
                revenue_after_sum += row['revenue_after'] 
            
            for ind, row in enumerate(dataset_list): 

                row['receipts_flats_sum'] = receipts_flats_sum
                row['receipts_nonresidental_sum'] = receipts_nonresidental_sum                                   
                row['receipts_parking_sum'] = receipts_parking_sum   

                row['advertising_sum'] = advertising_sum
                row['commercial_staff_sum'] = commercial_staff_sum                                   
                row['remuneration_to_banks_sum'] = remuneration_to_banks_sum   
                row['adm_pers_sum'] = adm_pers_sum
                row['rent_sum'] = rent_sum
                row['land_tax_sum'] = land_tax_sum
                row['vat_nonresidental_sum'] = vat_nonresidental_sum
                row['vat_parking_sum'] = vat_parking_sum

                row['cost_price_sum'] = cost_price_sum

                row['bridge_payout_sum'] = bridge_payout_sum
                row['escrow_payout_sum'] = escrow_payout_sum

                row['entrance_rights_sum'] = entrance_rights_sum                
                row['escrow_disclosure_sum'] = escrow_disclosure_sum
                row['revenue_after_sum'] = revenue_after_sum                

            for ind, row in enumerate(dataset_list):
            
                row['taxes_from_gross_profit'] = self.get_taxes_from_gross_profit(row, row_prev)
                row['realization_costs'] += row['taxes_from_gross_profit']

                row['project_expences'] = self.get_project_expenses(row, row_prev)
                row['net_profit'] = self.get_net_profit(row, row_prev)

        dataset = pd.DataFrame(dataset_list)

        if field:
            result = dataset['field']
        else:
            result = dataset
        return result


class DirectModel:

    def __init__(self, cb_model=None, cb_ratio = 0.2):
        self.processor = ProcessorRec()
        self.cb_model = cb_model
        self.cb_ratio = cb_ratio
        self.model_bid = False

    def _get_model_dataset(self):
        periods = list(range(136))

        dataset =  pd.DataFrame({'period': periods})

        dataset['sale_start_period_q1'] = self._date_to_period(datetime(2025, 10, 1))
        dataset['sale_start_period_q2'] = self._date_to_period(datetime(2025, 10, 1))

        dataset['temp_build_q1'] = 40
        dataset['temp_build_q2'] = 32

        dataset['adm_percent_q1'] = 0.049
        dataset['adm_percent_q2'] = 0.049

        dataset['smr_flats_pc_q1'] = 60
        dataset['smr_flats_pc_q2'] = 60

        dataset['price_grows_q1'] = 0.164912280701758
        dataset['price_grows_q2'] = 0.164912280701758

        dataset['start_price_flats_q1'] = 270000
        dataset['start_price_flats_q2'] = 270000

        dataset['start_price_nonresidental_q1'] = 240000
        dataset['start_price_nonresidental_q2'] = 240000

        dataset['start_price_parking_q1'] = 1500000
        dataset['start_price_parking_q2'] = 1500000

        dataset['key_bid_0'] = 0.21
        dataset['key_bid_1'] = 0.21
        dataset['key_bid_2'] = 0.21
        dataset['key_bid_3'] = 0.21
        dataset['key_bid_4'] = 0.21

        return dataset

    def _date_to_period(self, p_date):
        n_year = 2021
        n_month = (4-1)*3+1

        years = p_date.year - n_year
        quaters = math.ceil((p_date.month - n_month)/3)

        return years*4 + quaters

    def _get_x_columns(self, inner=False):
        if not inner:
            if hasattr(self, 'model_bid') and self.model_bid:
                return ['ind_0_an_0_num_1',
                    'ind_0_an_1_num_1',
                    'ind_1_an_0_num_0',
                    'ind_1_an_1_num_0',
                    'ind_2_an_0_num_1',
                    'ind_2_an_1_num_1',
                    'ind_2_an_2_num_1',
                    'ind_2_an_3_num_1',
                    'ind_2_an_4_num_1',
                    'ind_2_an_5_num_1',
                    'ind_3_an_0_num_1',
                    'ind_3_an_1_num_1',
                    'ind_4_an_0_num_1',
                    'ind_4_an_1_num_1',
                    'ind_5_num_1',
                    'ind_6_num_1',
                    'ind_7_num_1',
                    'ind_8_num_1',
                    'ind_9_num_1',
                    'ind_10_num_1',
                    'period_number',
                    'ind_5_num_1_1']
            else:
                return ['ind_0_an_0_num_1',
                'ind_0_an_1_num_1',
                'ind_1_an_0_num_0',
                'ind_1_an_1_num_0',
                'ind_2_an_0_num_1',
                'ind_2_an_1_num_1',
                'ind_2_an_2_num_1',
                'ind_2_an_3_num_1',
                'ind_2_an_4_num_1',
                'ind_2_an_5_num_1',
                'ind_3_an_0_num_1',
                'ind_3_an_1_num_1',
                'ind_4_num_1',
                'ind_5_an_0_num_1',
                'ind_5_an_1_num_1',
                'ind_6_num_1',
                'period_number',
                'ind_4_num_1_1',
                'ind_4_num_1_2',
                'ind_4_num_1_3',
                'ind_4_num_1_4',
                'ind_6_num_1_1']
        else:
            if hasattr(self, 'model_bid') and self.model_bid: 
                return  ['sale_start_period_q2',
                        'sale_start_period_q1',
                        'temp_build_q2',
                        'temp_build_q1',
                        'start_price_parking_q2',
                        'start_price_parking_q1',
                        'start_price_nonresidental_q2',
                        'start_price_flats_q2',
                        'start_price_nonresidental_q1',
                        'start_price_flats_q1',
                        'smr_flats_pc_q2',
                        'smr_flats_pc_q1',
                        'price_grows_q2',
                        'price_grows_q1',
                        'adm_percent_q1',
                        'key_bid_0',
                        'key_bid_1',
                        'key_bid_2',
                        'key_bid_3',
                        'key_bid_4',
                        'period',
                        'adm_percent_q2']
            else:           
                return ['sale_start_period_q2',
                    'sale_start_period_q1',
                    'temp_build_q2',
                    'temp_build_q1',
                    'start_price_parking_q2',
                    'start_price_parking_q1',
                    'start_price_nonresidental_q2',
                    'start_price_flats_q2',
                    'start_price_nonresidental_q1',
                    'start_price_flats_q1',
                    'smr_flats_pc_q2',
                    'smr_flats_pc_q1',
                    'key_bid_0',
                    'price_grows_q2',
                    'price_grows_q1',
                    'adm_percent_q1',
                    'period',
                    'key_bid_1',
                    'key_bid_2',
                    'key_bid_3',
                    'key_bid_4',
                    'adm_percent_q2']

    def _get_columns_hash(self, row, columns):
        values_list = []
        for col in columns:
            values_list.append(row[col])

        hash_object = hashlib.md5(str(values_list).encode())
        hex_hash = hash_object.hexdigest()

        return hex_hash

    def _get_batches(self, X_pd):
        inner_columns = [el for el in self._get_x_columns(inner=True) if el != 'period']

        X_pd['columns_hash'] = X_pd.apply(lambda x: self._get_columns_hash(x, columns=inner_columns), axis=1)

        batches = []
        batch_hashes = list(X_pd['columns_hash'].unique())
        for batch_hash in batch_hashes:
            X_batch = X_pd.loc[X_pd['columns_hash'] == batch_hash]
            batches.append(X_batch)

        return batches          

    def predict(self, X):
        model_dataset = self._get_model_dataset()

        pd_x = pd.DataFrame(X, columns=self._get_x_columns())
        if not (hasattr(self, 'model_bid') and self.model_bid):
            key_bid_cols = ['ind_5_an_0_num_1',
                        'ind_5_an_1_num_1',
                        'ind_6_num_1',
                        'ind_4_num_1',
                        'ind_4_num_1_1',
                        'ind_4_num_1_2',
                        'ind_4_num_1_3',
                        'ind_4_num_1_4',
                        'ind_6_num_1_1']
        else:
            key_bid_cols = ['ind_4_an_0_num_1',
                            'ind_4_an_1_num_1',
                            'ind_5_num_1',
                            'ind_6_num_1',
                            'ind_7_num_1',
                            'ind_8_num_1',
                            'ind_9_num_1',
                            'ind_10_num_1',
                            'ind_5_num_1_1']

        for col in key_bid_cols:
            pd_x[col] = pd_x[col]/100

        pd_x['ind_3_an_0_num_1'] = pd_x['ind_3_an_0_num_1']/1000
        pd_x['ind_3_an_1_num_1'] = pd_x['ind_3_an_1_num_1']/1000               

        to_rename = dict(zip(self._get_x_columns(), self._get_x_columns(inner=True)))
        pd_x = pd_x.rename(to_rename, axis=1)

        x_batches = self._get_batches(pd_x)

        value_column_names = [el for el in self._get_x_columns(inner=True) if el != 'period']
        result_batches = []
        for x_batch in x_batches:
            c_x_batch = x_batch.copy()
            c_x_batch['sale_start_period_q1'] = c_x_batch['sale_start_period_q1'].astype('int')
            c_x_batch['sale_start_period_q2'] = c_x_batch['sale_start_period_q2'].astype('int')            

            c_model_dataset = model_dataset.copy()

            col_values = {el: c_x_batch[el].unique()[0] for el in value_column_names}

            for col_name, col_value in col_values.items():
                c_model_dataset[col_name] = col_value

            cc_model_dataset = self.processor.calculate_ds(c_model_dataset)
            c_model_dataset['net_profit'] = cc_model_dataset['net_profit']
            c_x_batch = c_x_batch.merge(c_model_dataset[['period', 'net_profit']], on='period', how='left')
            c_x_batch = c_x_batch.fillna(0)
            result_batches.append(c_x_batch)
        
        X = pd.concat(result_batches, axis=0)
        X = X.drop('columns_hash', axis=1)

        X = X.sort_index()
        X['net_profit'] = X['net_profit']*1000
        result = X['net_profit'].to_numpy()

        if self.cb_model:
            pd_x = pd.DataFrame(X, columns=self._get_x_columns())
            y_cb = self.cb_model.predict(pd_x.to_numpy())
        
            result_df = pd.DataFrame(result, columns=['y'])
            result_df['y_cb'] = y_cb

            result_df['result'] = self.cb_ratio*result_df['y_cb'] + (1-self.cb_ratio)*result_df['y']

            result = result_df['result'].to_numpy()

        return result
            
    def predict_raw(self, X):
        model_dataset = self._get_model_dataset()

        pd_x = pd.DataFrame(X, columns=self._get_x_columns(inner=True))

        result_list = []
        for ind, row in pd_x.iterrows():
            c_model_dataset = model_dataset.copy()
            for col in self._get_x_columns(inner=True):
                if col=='period':
                    continue
                c_model_dataset[col] = row[col]
            
            cc_model_dataset = self.processor.calculate_ds(c_model_dataset)
            results = cc_model_dataset.loc[cc_model_dataset['period'] == row['period']]['net_profit'].values
            result = results[0] if results else 0
            result_list.append(result*1000)

        result = np.array(result_list)

        return result   