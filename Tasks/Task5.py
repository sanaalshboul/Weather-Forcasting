# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:57:30 2019

@author: Administrator
"""
import re
import csv
from bs4 import BeautifulSoup
#from HTMLParser import HTMLParser
from html.parser import HTMLParser

#class MyHTMLParser(HTMLParser):
#    def handle_starttag(self,tag,attrs):
#        print('encountered a start tags: ',tag )
#    
#    def handle_endtag(self,tag):
#        print('encounter an end tag: ', tag)
#        
#    def handle_data(self, data):
#        print('encountered some data: ', data)
#        
#with open('C:\\Users\\Administrator.Omnisec421_05\\.spyder-py3\\RE-3.txt', encoding='utf8', errors='ignore') as file:
#    soup=file.read()
#    parser=MyHTMLParser()
#    parser.feed(soup)
    
number_elements=0
empty_tags=0
end_tags_num=0
start_tags_num=0
poemCounter=0
data=[]
urlCounter=0
urlList=[]
with open('RE-3.txt', encoding='utf8', errors='ignors') as file:
    lines=file.readlines()
    for line in lines:
        line=str(line)
        start_tags=re.findall('<ink|<script|<noscript|<img|<style|<body|<iframe|<div|<p|<a|<section|<form|<input|<meta|<nav|<ul|<li|<span|<h3|<main|<article|<header|<strong|<hr|<i|<em|<label|<select|<option|<footer|<h4', line)
        start_tags_num += len(start_tags)
        
        empty=re.findall('<br', line)
        empty_tags +=len(empty)
        
        end_tags=re.findall('</ink|</script|</noscript|</img|</style|</body|</iframe|</div|</p|</a|</section|</form|</input|</meta|</nav|</ul|</li|</span|</h3|</main|</article|</header|</strong|</hr|</i|</em|</label|</select|</option|</footer|</h4', line)
        end_tags_num += len(end_tags)
        
        poem=re.findall('poem', line)
        poemCounter +=len(poem)
        if re.findall('^<title>(.+?)</title>',line):
            data.append(re.findall('^<title>(.+?)</title>',line))
            
        url=re.findall("src | srcset | href", line)
        urlCounter +=len(url)
        
        if re.findall("src",line):
            urlList.append(re.findall("src='(.+?)'",line))
            urlList.append(re.findall("srcset='(.+?)'",line))
        
        urlList.append(re.findall("href='(.+?)'", line))
            
print('The number of elements in the file: ',min(start_tags_num, end_tags_num))
print('The number of empty tags in the file: ', empty_tags)
print('The total number of tags in the file: ', empty_tags+start_tags_num+end_tags_num )
print('The number of poem words in the file: ', poemCounter)
print('The content of the title (without the tags): ', data)
print('The number of URLs in the file: ', urlCounter)
print('List of all URLs in the file: ', urlList)
###############################################################################
###############################################################################
###############################################################################
##mission 2 part 1,2
#key_words=[]
#goodCounter=0
#funCounter=0
#greatCounter=0
#badCounter=0
#sadCounter=0
#suckCounter=0
#with open ('C:\\Users\\Administrator.Omnisec421_05\\.spyder-py3\\short_movies.csv', encoding='utf8', errors='ignore') as file:
#    reader=csv.DictReader(file)
#    for i in reader:
#        temp=i['text'].split(" ")
#        for j in temp:
#            if j=="good":
#                goodCounter +=1
#            elif j=="funny" or j== "fun":
#                funCounter +=1
#            elif j== "great":
#                greatCounter +=1
#            elif j=="bad":
#                badCounter +=1
#            elif j== "sad":
#                sadCounter +=1
#            elif j== "suck" or j=="sucks":
#                suckCounter +=1
#                
#key_words.append(goodCounter)
#key_words.append(funCounter)
#key_words.append(greatCounter)
#key_words.append(badCounter)
#key_words.append(sadCounter)
#key_words.append(suckCounter)
#
#print("The accurance number of good, fun(ny), and great word in the file recpectively are: ", key_words[0:3] )
#print("The accurance number of bad, sad, and suck words in the file respectively are: ",key_words[3:6] )
#
################################################################################
##mission 2 part 3, 4, 5, 6
#counter=0
#counterGreat=0
#counterBad=0
#list_html_reviews=[]
#list_html_good=[]
#list_html_bad=[]
#list_html_ratio=[]
#with open('C:\\Users\\Administrator.Omnisec421_05\\.spyder-py3\\short_movies.csv', encoding='utf8', errors='ignors') as file:
#    reader=csv.DictReader(file)
#    for i in reader:
#        temp=i['text'].split(" ")
#        for j in temp:
#            if j== "good" or j=="funny" or j== "fun" or j== "great" or j=="bad" or j== "sad" or j== "suck" or j=="sucks":
#                counter +=1
#            if j== "good" or j=="funny" or j== "fun" or j== "great":
#                counterGreat +=1
#            if  j=="bad" or j== "sad" or j== "suck" or j=="sucks":
#                counterBad +=1
#                
#        list_html_reviews.append((int(i['html']), counter))
#        counter=0
#        list_html_good.append((int(i['html']),counterGreat))
#        counterGreat=0
#        list_html_bad.append((int(i['html']), counterBad))
#        counterBad=0
#        list_html_ratio.append(counterGreat/counterBad)
#
#list_html_reviews=list(sorted(list_html_reviews, key=lambda x: x[1], reverse=True))   
#list_html_good=list(sorted(list_html_good, key=lambda x: x[1], reverse=True)) 
#list_html_bad=list(sorted(list_html_bad, key=lambda x: x[1], reverse=True))
#
#print("The html with biggest number of reviews:", list_html_reviews[0][0])
#print("The html with the biggest number of occurrences of the words good, fun(ny), and great in its review:", list_html_good[0][0])
#print("The html with the biggest number of occurrences of the words bad, sad, and suck(s) in its review:", list_html_bad[0][0])
#print('A list represents the ratio between good reviews and bad reviews for each html: ',list_html_ratio)
#
################################################################################
################################################################################
################################################################################
##mission 3 part1,2
#users_mention=[]
#loveCounter=0
#hateCounter=0
#with open('C:\\Users\\Administrator.Omnisec421_05\\.spyder-py3\\short_tweets.csv', encoding='utf8', errors='ignors') as file:
#    reader=csv.DictReader(file)
#    for i in reader:
#        temp=i['text'].split(' ')
#        
#        for j in temp:
#            if '@' in j and len(j)>1:
#                users_mention.append(j)
#            if 'love' in j or 'Love' in j:
#                loveCounter +=1
#            if 'hate' in j or 'Hate' in j:
#                hateCounter +=1
#
#if loveCounter > hateCounter:           
#    print("Number of love more than hate")
#else: print('Number of hate more than love')
################################################################################
##mission 3 part3
#list_hate=[]
#with open('C:\\Users\\Administrator.Omnisec421_05\\.spyder-py3\\short_tweets.csv', encoding='utf8', errors='ignors') as file:
#    reader=csv.DictReader(file)
#    for i in reader:
#        temp=i['text'].split(' ')
#        for j in range(len(temp)):
#            if j> len(temp)-3:
#                break
#            elif temp[j]=="hate" or temp[j]=="Hate":
#               #print(temp[j], temp[j+1], temp[j+2])
#               list_hate.append((temp[j], temp[j+1], temp[j+2])) 
#print(list_hate)
################################################################################
##mission 3 part4
#list_love=[]
#with open('C:\\Users\\Administrator.Omnisec421_05\\.spyder-py3\\short_tweets.csv', encoding='utf8', errors='ignors') as file:
#    reader=csv.DictReader(file)
#    for i in reader:
#        temp=i['text'].split(' ')
#        for j in range(len(temp)):
#            if j<2:
#                continue
#            elif temp[j]=="love" or temp[j]=="Love":
#               #print(temp[j], temp[j+1], temp[j+2])
#               list_love.append((temp[j-1], temp[j-2],temp[j])) 
#print(list_love)