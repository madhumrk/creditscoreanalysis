#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

import time, warnings
import datetime as dt

#visualizations
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

warnings.filterwarnings("ignore")


# In[2]:


retail_df = pd.read_excel(r"C:\Users\madhu\Downloads\CreditAnalysis_data.xlsx")


# In[3]:


retail_df.head()


# In[4]:


retail_df.drop(['master_order_id','master_order_status'],inplace=True,axis = 1)


# In[7]:


#!pip install dataprep


# In[5]:


from dataprep.eda import create_report


# In[6]:


create_report(retail_df)


# In[8]:


retail_df.columns ## treating missing values 


# In[9]:


#remove rows where customerID are NA
retail_df.dropna(subset=['ordereditem_product_id'],how='all',inplace=True)
retail_df.shape


# In[19]:


###retail_df['created'] = retail_df['created'].dt.strftime('%Y-%m-%d')


# In[20]:


retail_df.created = retail_df.created.astype('str')
retail_df.dtypes


# In[33]:


#restrict the data to one full year because it's better to use a metric per Months or Years in RFM
retail_df = retail_df[retail_df['created']>= "2018-3-31"]
retail_df.shape


# # RFM Analysis
# 
# 
# RFM (Recency, Frequency, Monetary) analysis is a customer segmentation technique that uses past purchase behavior to divide customers into groups. RFM helps divide customers into various categories or clusters to identify customers who are more likely to respond to promotions and also for future personalization services.
# 
# RECENCY (R): Days since last purchase
# FREQUENCY (F): Total number of purchases
# MONETARY VALUE (M): Total money this customer spent.
# We will create those 3 customer attributes for each customer.

# # Recency
# 
# To calculate recency, we need to choose a date point from which we evaluate how many days ago was the retailers's last order.

# In[31]:


#last date available in our dataset
retail_df['created'].min()


# In[24]:


#create a new column called date which contains the date of invoice only
retail_df['date'] = pd.DatetimeIndex(retail_df['created']).date


# In[25]:


retail_df.head()


# In[38]:


#group by customers and check last date of purshace
recency_df = retail_df.groupby(by='retailer_names', as_index=False)['date'].max()
recency_df.columns = ['Retailernames','LastPurshaceDate']
recency_df.head()


# In[27]:


now = dt.date(2018,3,31)
print(now)


# In[39]:


#calculate recency
recency_df['Recency'] = recency_df['LastPurshaceDate'].apply(lambda x: (now - x).days)


# In[40]:


recency_df.head()


# In[41]:


recency_df.tail()


# In[42]:


#drop LastPurchaseDate as we don't need it anymore
recency_df.drop('LastPurshaceDate',axis=1,inplace=True)


# # Frequency
# 
# 
# Frequency helps us to know how many times a customer purchased from us. To do that we need to check how many orders are registered by the same retailers.

# In[45]:


# drop duplicates
retail_df_copy = retail_df
retail_df_copy.drop_duplicates(subset=['order_id', 'retailer_names'], keep="first", inplace=True)
#calculate frequency of purchases
frequency_df = retail_df_copy.groupby(by=['retailer_names'], as_index=False)['order_id'].count()
frequency_df.columns = ['Retailernames','Frequency']
frequency_df.head()


# # Monetary
# 
# 
# Monetary attribute answers the question: How much money did the retailer spent over time?
# 
# To do that, first, we will create a new column total cost to have the total price per order.

# In[46]:


#create column total cost
retail_df['TotalCost'] = retail_df['ordereditem_quantity'] * retail_df['ordereditem_unit_price_net']


# In[47]:


monetary_df = retail_df.groupby(by='retailer_names',as_index=False).agg({'TotalCost': 'sum'})
monetary_df.columns = ['Retailernames','Monetary']
monetary_df.head()


# # Create RFM Table

# In[50]:


#merge recency dataframe with frequency dataframe
temp_df = recency_df.merge(frequency_df,on='Retailernames')
temp_df.head()


# In[51]:


#merge with monetary dataframe to get a table with the 3 columns
rfm_df = temp_df.merge(monetary_df,on='Retailernames')
#use CustomerID as index
rfm_df.set_index('Retailernames',inplace=True)
#check the head
rfm_df.head()


# # RFM Table Correctness verification

# In[53]:


retail_df[retail_df['retailer_names']=='RetailerID1']


# In[54]:


#RFM Quartiles
quantiles = rfm_df.quantile(q=[0.25,0.5,0.75])
quantiles


# In[55]:


quantiles.to_dict()


# # Creation of RFM Segments
# 
# We will create two segmentation classes since, high recency is bad, while high frequency and monetary value is good.

# In[56]:


# Arguments (x = value, p = recency, monetary_value, frequency, d = quartiles dict)
def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4


# In[57]:


#create rfm segmentation table
rfm_segmentation = rfm_df
rfm_segmentation['R_Quartile'] = rfm_segmentation['Recency'].apply(RScore, args=('Recency',quantiles,))
rfm_segmentation['F_Quartile'] = rfm_segmentation['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
rfm_segmentation['M_Quartile'] = rfm_segmentation['Monetary'].apply(FMScore, args=('Monetary',quantiles,))


# In[58]:


rfm_segmentation.head()


# Now that we have the score of each customer, we can represent our customer segmentation. First, we need to combine the scores (R_Quartile, F_Quartile,M_Quartile) together.

# In[60]:


rfm_segmentation['RFMScore'] = rfm_segmentation.R_Quartile.map(str)                             + rfm_segmentation.F_Quartile.map(str)                             + rfm_segmentation.M_Quartile.map(str)
rfm_segmentation.head()


# In[61]:


rfm_segmentation.tail()


# Best Recency score = 4: most recently purchase. Best Frequency score = 4: most quantity purchase. Best Monetary score = 4: spent the most.
# 
# Let's see who are our Champions (best retailers).

# In[62]:


rfm_segmentation[rfm_segmentation['RFMScore']=='444'].sort_values('Monetary', ascending=False).head(10)


# In[63]:


print("Best Retailers: ",len(rfm_segmentation[rfm_segmentation['RFMScore']=='444']))
print('Loyal Retailers: ',len(rfm_segmentation[rfm_segmentation['F_Quartile']==4]))
print("Big Spenders: ",len(rfm_segmentation[rfm_segmentation['M_Quartile']==4]))
print('Almost Lost: ', len(rfm_segmentation[rfm_segmentation['RFMScore']=='244']))
print('Lost Retailers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='144']))
print('Lost Cheap Retailers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='111']))


# In[ ]:




