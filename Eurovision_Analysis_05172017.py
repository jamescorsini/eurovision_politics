
# coding: utf-8

# ## Data Robot Eurovision Analysis

# *My Question* Is there a political bias with Eurovision voting?
# 
# *What is it?* Eurovision is an international (mostly European) song contest that has been happening since 1956 (more information: https://en.wikipedia.org/wiki/Eurovision_Song_Contest)
# 
# *How will this be analyzed?* This analysis will go through the open dataset https://eurovision.tv/history/full-split-results which provides all of the points given by country and how they were awarded.  I will first clean the data and then organize it into two sets, one of which will be how many points a country gave to it's neighboring countries and one set with all the other points.
# 

# In[1]:

# Importing 
import pandas as pd
import numpy as np
import scipy.stats as stats

# For my stylish plots
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().magic(u'pylab inline')


# In[2]:

# Import data into df
gf_2017 = pd.read_excel('ESC2017_GF_Results.xlsx',sheetname='FinalResults')
# Verify data is there
gf_2017.head()


# In[3]:

# Dropping first couple cols (because who needs Rank, indentifier and total)
# Also, renaming index to Countries because it's a lot easier and makes more sense this way
points_df = gf_2017.copy()
points_df.index = points_df['Country']
del points_df['Rank']
del points_df['DDI']
del points_df['Country']
del points_df['Total']


# In[4]:

# Using simple measure for now to determine bias: there is political bias when countries vote for neighboring countries 
# more than others
# TODO: key improvement here is to dig into the political relationaships and get a better 'neighbor' definition.
# Currently this leaves out countries like Australia and Israel

# Building dict of neighboring countries 
neighbors_dict = {
    u'Portugal':['Spain'],
    u'Bulgaria':['Romania','Greece','Macedonia','Serbia'], 
    u'Moldova':['Romania','Ukraine'], 
    u'Belgium':['Netherlands','France','Germany'], 
    u'Sweden':['Norway','Denmark','Finland'],
    u'Italy':['Croatia','Austria','SanMarino','Switzerland','Slovenia','Malta'], 
    u'Romania':['Moldova','Bulgaria','Hungary','Ukraine','Serbia'], 
    u'Hungary':['Ukraine','Romania','Austria','Croatia'], 
    u'Australia':[], 
    u'Norway':['Sweden','Denmark','Finland'],
    u'Netherlands':['Belgium','Germany'], 
    u'France':['Spain','Belgium','Switzerland'], 
    u'Croatia':['Hungary','Italy','Slovenia','Montenegro'], 
    u'Azerbaijan':['Armenia','Georgia'],
    u'UnitedKingdom':[], 
    u'Austria':['Italy','Germany','Hungary','Slovenia','CzechRepublic'], 
    u'Belarus':['Ukraine','Poland','Latvia','Lithuania'], 
    u'Armenia':['Azerbaijan','Georgia'], 
    u'Greece':['Cyprus','Bulgaria','Macedonia','Albania'],
    u'Denmark':['Sweeden','Norway','Finland'], 
    u'Cyprus':['Greece'], 
    u'Poland':['Germany','Ukraine','Belarus','Lithuania','CzechRepublic'], 
    u'Israel':[], 
    u'Ukraine':['Belarus','Moldova','Hungary','Poland'], 
    u'Germany':['Netherlands','Poland','Austria','Belgium','France','Switzerland','CzechRepublic'],
    u'Spain':['Portugal','France']    
}


# In[5]:

# Creates neighbor_df which is a sparse binary df identifying neighbors (will be data mask moving forward)
neighbor_df = pd.DataFrame(columns=points_df.columns, index=points_df.index)
neighbor_df = neighbor_df.fillna(0)

for country in neighbors_dict.keys():
    for neighbor in neighbors_dict[country]:
        neighbor_df[country][neighbor] = 1


# In[6]:

# Creates compare_df wihch has average points given to neighboring countries and non neighboring countries by country
# Note: lots of NaNs here because only the countries in the final round of scoring (24 of them) where used
compare_df = pd.DataFrame(index=points_df.columns,columns=['Neighbors Avg','Non Neighbors Avg'])
compare_df['Neighbors Avg'] = points_df[neighbor_df==1].mean()
compare_df['Non Neighbors Avg'] = points_df[neighbor_df!=1].mean()

compare_df.head()


# In[12]:

# Export data
compare_df.to_csv('comparison_df.csv')


# In[7]:

# A looks at our data
compare_df.dropna()


# In[8]:

# Some basic stats on our data
compare_df.describe()


# In[9]:

# Creating a neighbor multiple.  This shows how much more a country voted for it's neighbors opposed to others
# Also, saved it to csv for use in tableau viz
neighbor_mult = (compare_df['Neighbors Avg']/compare_df['Non Neighbors Avg'])
neighbor_mult.to_csv('neighbor_mult.csv')

neighbor_mult.dropna()


# ### Viz of the data

# In[10]:

plt.figure()

h = compare_df['Neighbors Avg']
h = h.sort_values()
fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
plt.plot(h,fit, color='blue')

h = compare_df['Non Neighbors Avg']
h = h.sort_values()
fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

plt.plot(h,fit, color='red')
plt.title('Neighbors comparision')
plt.xlabel("% Time Difference")

plt.figure()
compare_df['Neighbors Avg'].hist()
compare_df['Non Neighbors Avg'].hist()
plt.title('Histograms')


# ### T-Test Time

# In[11]:

# Method now is using Student T-Test to determine if there is a significant difference between the countries voting 
# for their neighbors and the conutries when they voted for other non-neighbors

# Null hypothesis: That there is no difference between these sets

a = compare_df['Neighbors Avg']
b = compare_df['Non Neighbors Avg']

stat, pval = stats.ttest_rel(a,b,nan_policy='omit')
print 'pval: %0.2f tstat: %2.2f' % (pval, stat)


# A p values of less than 0.05 here indicates that we can reject the null hypothesis that says they are the same and conclude that they are in fact statistically and significantly diffent.  Which leads us back to the original question, is there political bias?  And the answer would be yes

# ### Conclusion/ Areas to Improve

# *Conclusion:* From this analysis we can conclude that **there is a bias** when it comes to Eurovision countries and their voting patterns
# 
# *Areas to improve:* There are some definite areas to improve this analysis given some more time  
# - More data.  This would confirm the trend (over multiple years) as well as analyze the countries that weren't accounted for in this data set
# - Better test metric.  Right now it's simply neighbors but there are some relationships that are not represented with this.  For example, Armenia and Azerbaijan extremely dislike each other and is skewing this analysis because they are neighbors
