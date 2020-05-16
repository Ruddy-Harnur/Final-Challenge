import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession

import pyspark.sql.functions as f 
from datetime import datetime
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.sql.window import Window
from pyspark.sql.functions import lit
from pyspark.sql.functions import lower, col
from pyspark.sql import types as t
from pyspark.sql.types import IntegerType

from itertools import chain
from pyspark.sql.functions import col, create_map, lit

import numpy as np
#spark.debug.maxToStringFields=100
#import statsmodels.api as sm

#import statsmodels.formula.api as smf

# from pyspark.sql.functions import regexp_replace, col
# from pyspark.ml.regression import LinearRegression
# from sklearn.linear_model import LinearRegression
# from pyspark.sql.functions import broadcast

from pyspark.sql.functions import *

if __name__=='__main__':
    sc = SparkContext()
    spark = SparkSession(sc)
    pv = spark.read.csv('hdfs:///tmp/bdm/nyc_parking_violation/', header = True,inferSchema = True)
    pv = pv.select('Issue Date', 'Violation County', 'Street Name', 'House Number')
    pv = pv.withColumn('Date', f.to_date('Issue Date'))#.alias('Year')
    pv.show()
    pv = pv.withColumn('Year',f.year(pv['Date']))
    pv.show()
    pv.dtypes)
    pv = pv.filter(pv["Year"] >= (2015)) \
       .filter(pv["Year"] <= lit(2019))
    pv.show()
    #pv = pv.select(f.year(pv['Year']))
    pv = pv.na.drop()
    pv.show()
    pv = pv.withColumn('street name',f.lower(pv['Street Name']))
    pv.show()
    #pv = pv.select('Year', (to_date('Year', 'MM/dd/yyyy')))
    #dates = (2015, 2019)
    #pv = pv.where(col('Year').between(*dates))
    #pv = pv.filter(pv['Year']2015 & pv['Year']<=2019)
    #pv = pv.withColumn('Year',(pv['Issue Date'].substr(-4,4)))
    #pv = pv.filter(pv['Year']=='2015' | pv['Year']=='2016' | pv['Year']=='2017' | pv['Year']=='2018' | pv['Year']=='2019')
    #pv = pv.filter(
    
    borough_dict = {'NY':1, 'MAN':1, 'MH':1, 'NEWY':1, 'NEW':1, 'Y':1, "NY":1,
                'BX':2, 'BRONX':2,
                'K':3, 'BK':3, 'KING':3, 'KINGS':3,
                'Q':4, 'QN':4, 'QNS':4, 'QU':4, 'QUEEN':4,
                'R':5, 'RICHMOND':5}
    mapping_expr = create_map([lit(x) for x in chain(*borough_dict.items())])
    pv = pv.withColumn("BOROCODE", mapping_expr.getItem(col("Violation County")))
    pv = pv.withColumn("HN_int",(f.regexp_replace("House Number", "-", "")))
    pv = pv.withColumn("HN_int",regexp_replace(col("HN_int"), " ", ""))
    pv = pv.withColumn("HN_int", pv["HN_int"].cast(IntegerType()))
    pv = pv.na.drop() \
                     .select('Year','BOROCODE', 'street name', 'HN_int')
    
    df_centerline = spark.read.csv('hdfs:///tmp/bdm/nyc_cscl.csv', header = True, inferSchema = True)
    df_centerline = df_centerline.select('PHYSICALID', 'ST_LABEL','FULL_STREE', 'BOROCODE', 'L_LOW_HN', 'L_HIGH_HN', 'R_LOW_HN', 'R_HIGH_HN')
    
    df_centerline = df_centerline.withColumn("L_LOW_int",(f.regexp_replace("L_LOW_HN", "-", "")))
    df_centerline = df_centerline.withColumn("L_LOW_int",regexp_replace(col("L_LOW_int"), " ", ""))
    df_centerline = df_centerline.withColumn("L_LOW_int", df_centerline["L_LOW_int"].cast(IntegerType()))
    df_centerline = df_centerline.withColumn("L_HIGH_int",(f.regexp_replace("L_HIGH_HN", "-", "")))
    df_centerline = df_centerline.withColumn("L_HIGH_int",regexp_replace(col("L_HIGH_int"), " ", ""))
    df_centerline = df_centerline.withColumn("L_HIGH_int", df_centerline["L_HIGH_int"].cast(IntegerType()))
    df_centerline = df_centerline.withColumn("R_LOW_int",(f.regexp_replace("R_LOW_HN", "-", "")))
    df_centerline = df_centerline.withColumn("R_LOW_int",regexp_replace(col("R_LOW_int"), " ", ""))
    df_centerline = df_centerline.withColumn("R_LOW_int", df_centerline["R_LOW_int"].cast(IntegerType()))
    df_centerline = df_centerline.withColumn("R_HIGH_int",(f.regexp_replace("R_HIGH_HN", "-", "")))
    df_centerline = df_centerline.withColumn("R_HIGH_int",regexp_replace(col("R_HIGH_int"), " ", ""))
    df_centerline = df_centerline.withColumn("R_HIGH_int", df_centerline["R_HIGH_int"].cast(IntegerType()))
    
    df_centerline = df_centerline.select('PHYSICALID', 'ST_LABEL', 'FULL_STREE', 'BOROCODE', 
                                     'L_LOW_int', 'L_HIGH_int', 'R_LOW_int', 'R_HIGH_int')
    df_centerline = df_centerline.withColumn('ST_LABEL', lower(col('ST_LABEL'))).withColumn('FULL_STREE', lower(col('FULL_STREE')))
    
    result_df = pv.join(broadcast(df_centerline),(pv["BOROCODE"]==df_centerline["BOROCODE"]) & 
                          ((pv["street name"] == df_centerline['ST_LABEL']) | (pv['street name'] == df_centerline['FULL_STREE'])) &
                          (((pv['HN_int']%2==1) & (pv['HN_int'] >= df_centerline['L_LOW_int']) & (pv['HN_int'] <= df_centerline['L_HIGH_int'])) |
                          ((pv['HN_int']%2==0) & (pv['HN_int'] >= df_centerline['R_LOW_int']) & (pv['HN_int'] <= df_centerline['R_HIGH_int']))))
                      
    
    pivoted = result_df.groupBy("PHYSICALID").pivot("YEAR",['2015','2016','2017','2018','2019']).count()
    
    final_df = pivoted.join(broadcast(df_centerline), ['PHYSICALID'], how='right')
    final_df = final_df.select('PHYSICALID', '2015', '2016', '2017', '2018', '2019')
    final_df = final_df.na.fill(0)
     
    
    # def slope(a, b, c, d, e):
        # X = ([2015, 2016, 2017, 2018, 2019])
        # X = sm.add_constant(X)
        # y = ([a, b, c, d, e])
        # model = sm.OLS(y,X)
        # #(y, X)
        # results = model.fit()
        # return((results.params[1]))
    
    # final_df = final_df.withColumn('OLS', slope(final_df['2015'], final_df['2016'], final_df['2017'], 
                                               # final_df['2018'], final_df['2019']))
    
    
    final_df = final_df.orderBy('PHYSICALID')
    final_df.write.csv('x')