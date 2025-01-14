# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:40:40 2024

@author: Hp
"""
###Movie
import bs4
from bs4 import BeautifulSoup as bs
import requests
link="https://www.imdb.com/title/tt0454921/reviews/?ref_=tt_ov_ql_2"
page=requests.get(link)
page
page.content

## now let us parse the html page
soup=bs(page.content,'html.parser')
print(soup.prettify())

## now let us scrap the contents
#Now let us try to identify the titles of reviews
title=soup.find_all('a',class_='title')
title
# when you will extract the web page got to all reviews rather top revews.when you
# click arrow icon and the total reviews ,there you will find span has no class
# you will have to go to parent icon i.e.a
#now let us extract the data
review_titles=[]
for i in range(0,len(title)):
    review_titles.append(title[i].get_text())
review_titles

# ouput we will get consists of \n 
review_titles[:]=[ title.strip('\n')for title in review_titles]
review_titles
len(review_titles)
#Got 24 review titles


##now let us scrap ratings
rating=soup.find_all('span',class_='rating-other-user-rating')
rating
###we got the data
rate=[]
for i in range(0,len(rating)):
    rate.append(rating[i].get_text())
rate
rate= [r.strip().split('/')[0] for r in rate]
rate
len(rate)
rate.append('')
rate.append('')
len(rate)
#Got 24 ratings


##now let us review body
review=soup.find_all('div',class_='text show-more__control')
review
###we got the data
review_body=[]
for i in range(0,len(review)):
    review_body.append(review[i].get_text())
review_body
review_body=[ reviews.strip('\n\n')for reviews in review_body]
len(review_body)

###convert to csv file
import pandas as pd
df=pd.DataFrame()
df['review_title']=review_titles
df['rate']=rate
df['review_body']=review_body
df
df.to_csv("C:\Assignments DS\Web Scrapping\TPOH_reviews.csv",index=True)
########################################################   
#sentiment analysis
import pandas as pd
from textblob import TextBlob
df=pd.read_csv("C:\Assignments DS\Web Scrapping\TPOH_reviews.csv")
df.head()
df['polarity']=df['review_body'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']



##Shoes
import bs4
from bs4 import BeautifulSoup as bs
import requests
link="https://www.amazon.in/Bacca-Bucci-Black-Orange-Running/dp/B08HNGK3B7"
page=requests.get(link)
page
page.content
## now let us parse the html page
soup=bs(page.content,'html.parser')
print(soup.prettify())
#when you parse HTML using BeautifulSoup, you are converting the 
#raw HTML content of a web page into a structured format, 
#like a tree, where you can easily locate and manipulate individual 
#elements (such as tags, attributes, or text).

#page.content=> provides the raw HTML content,
#while soup.prettify()=> offers a formatted, human-readable version of the parsed HTML content.

## now let us scrap the contents
names=soup.find_all('span',class_="a-profile-name")
names
### but the data contains with html tags,let us extract names from html tags
cust_names=[]
for i in range(0,len(names)):
    cust_names.append(names[i].get_text())
    
cust_names
len(cust_names)
cust_names.pop(-1)
cust_names.pop(-1)
cust_names.pop(-1)
cust_names.pop(-1)
len(cust_names)


### There are total 10 users names 
#Now let us try to identify the titles of reviews

title_rate=soup.find_all('a',class_='review-title')
tr_list = [x.text.strip() for x in title_rate]
tr_list
len(tr_list)
ratings = []
reviews = []

# Process each entry in tr_list
for i in tr_list:
    rating, review = i.split('\n', 1)
    ratings.append(rating)
    reviews.append(review)
ratings 
reviews 
rate = [int(i[0]) for i in ratings]
print(rate)
len(rate )
len(reviews )




## now let us scrap review body
reviews=soup.find_all("div",class_="a-row a-spacing-small review-data")
reviews
review_body=[]
for i in range(0,len(reviews)):
    review_body.append(reviews[i].get_text())
review_body
review_body=[ reviews.strip('\n\n')for reviews in review_body]
review_body
len(review_body)

##########################################
###convert to csv file
import pandas as pd
df=pd.DataFrame()
df['customer_names']=cust_names
df['review_title']=reviews
df['rate']=rate
df['review_body']=review_body
df
df.to_csv('C:\Assignments DS\Web Scrapping\Amazon_shoes_reviews.csv',index=True)
########################################################
#sentiment analysis
import pandas as pd
from textblob import TextBlob
df=pd.read_csv("C:\Assignments DS\Web Scrapping\Amazon_shoes_reviews.csv")
df.head()
df['polarity']=df['review_body'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity'] 


##Boat Eardopes
import bs4
from bs4 import BeautifulSoup as bs
import requests
link="https://www.boat-lifestyle.com/products/airdopes-alpha-true-wireless-earbuds?_gl=1*1asuzs9*_up*MQ..&gclid=CjwKCAjw0aS3BhA3EiwAKaD2ZUE1Z2Y1zY8L8vqS0r_ZywCCySZGLKmMMvpC4FI_5fZjG4ZTTBcMOBoCbO4QAvD_BwE"
page=requests.get(link)
page
page.content
## now let us parse the html page
soup=bs(page.content,'html.parser')
print(soup.prettify())
#when you parse HTML using BeautifulSoup, you are converting the 
#raw HTML content of a web page into a structured format, 
#like a tree, where you can easily locate and manipulate individual 
#elements (such as tags, attributes, or text).

#page.content=> provides the raw HTML content,
#while soup.prettify()=> offers a formatted, human-readable version of the parsed HTML content.

## now let us scrap the contents
names=soup.find_all('span',class_="jdgm-rev__author")
names
### but the data contains with html tags,let us extract names from html tags
cust_names=[]
for i in range(0,len(names)):
    cust_names.append(names[i].get_text())
    
cust_names
len(cust_names)
#cust_names.pop(-1)
#len(cust_names)


### There are total 6 users names 
#Now let us try to identify the titles of reviews

title=soup.find_all('b',class_="jdgm-rev__title")
title
# when you will extract the web page got to all reviews rather top revews.when you
# click arrow icon and the total reviews ,there you will find span has no class
# you will have to go to parent icon i.e.a
#now let us extract the data
review_titles=[]
for i in range(0,len(title)):
    review_titles.append(title[i].get_text())
review_titles

len(review_titles)
##now let us scrap ratings
rating=soup.find_all('span',class_="jdgm-rev__rating")
rating
###we got the data
ratings = [int(span['data-score']) for span in soup.find_all('span', {'class': 'jdgm-rev__rating'})]

# Print the ratings
print(ratings)

len(ratings)



## now let us scrap review body
reviews=soup.find_all("div",class_="jdgm-rev__body")
reviews
review_body=[]
for i in range(0,len(reviews)):
    review_body.append(reviews[i].get_text())
review_body
review_body=[ reviews.strip('\n\n')for reviews in review_body]
review_body
len(review_body)

##########################################
###convert to csv file
import pandas as pd
df=pd.DataFrame()
df['customer_names']=cust_names
df['review_title']=review_titles
df['rate']=ratings
df['review_body']=review_body
df
df.to_csv('C:\Assignments DS\Web Scrapping\Boat_ear_reviews.csv',index=True)
########################################################
#sentiment analysis
import pandas as pd
from textblob import TextBlob
df=pd.read_csv("C:\Assignments DS\Web Scrapping\Boat_ear_reviews.csv")
df.head()
df['polarity']=df['review_body'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity'] 





























































































