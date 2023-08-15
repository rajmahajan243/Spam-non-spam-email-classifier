# Assignment 3
# Name  - Raj Uday Mahajan
# Roll Number -  CS22M067
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import os

file=pd.read_csv('SMSSpamCollection',header=None, sep='\t',names=['Label', 'Email'])
print(file.shape)

test_path = "test"
os.chdir(test_path)

ds = file.sample(frac=1, random_state=1)
ds_len=len(ds)*0.8
index = round(ds_len)
train_data = ds[:index].reset_index(drop=True)
train_data['Email'] = train_data['Email'].str.replace('\W', ' ')
distinct_words = []
train_data['Email'] = train_data['Email'].str.lower()
train_data['Email'] = train_data['Email'].str.split()

for el in train_data['Email']:
  for wd in el:
    distinct_words.append(wd)

distinct_words = list(set(distinct_words))
WordCount = {word : [0]*len(train_data['Email'])
for word in distinct_words}

print(len(distinct_words))

for i,E_mail in enumerate(train_data['Email']):
  for wd in E_mail:
    WordCount[wd][i] =WordCount[wd][i]+ 1
WordCount = pd.DataFrame(WordCount)

cleaned_train = pd.concat([train_data, WordCount], axis=1)

spam = cleaned_train[cleaned_train['Label']=='spam']
p_spam = len(spam)/len(cleaned_train)
ham = cleaned_train[cleaned_train['Label']=='ham']
p_ham = len(ham)/len(cleaned_train)


num_spam = spam['Email'].apply(len).sum()
num_distinct_words = len(distinct_words)
num_ham = ham['Email'].apply(len).sum()


param_spam = {word:0 for word in distinct_words}
param_ham = {word:0 for word in distinct_words}
a = 1
for word in distinct_words:
  temp1=spam[word].sum() + a
  param_spam[word] =temp1 /(num_spam + a*num_distinct_words)
  temp2=ham[word].sum() + a
  param_ham[word] =temp2 /(num_ham + a*num_distinct_words)


def Ham_OR_Spam(ip_email):
   ip_email = re.sub('\W', ' ', ip_email)
   p_spam_ip_email = p_spam
   ip_email = ip_email.lower().split()
   p_ham_ip_email = p_ham

   for wd in ip_email:
      if wd in param_spam:
         p_spam_ip_email =p_spam_ip_email * param_spam[wd]
      if wd in param_ham:
         p_ham_ip_email =p_ham_ip_email * param_ham[wd]

   if p_ham_ip_email > p_spam_ip_email:
      return 'ham'
   else:
      return 'spam'




test_data = ds[index:].reset_index(drop=True)
test_data['prediction'] = test_data['Email'].apply(Ham_OR_Spam)
total = test_data.shape[0]
print(total)
count = 0
for row in test_data.iterrows():
   row = row[1]
   if row['prediction'] == row['Label'] :
      count =count + 1

Accuracy=count*100/total
print(count)
print('Accuracy:', Accuracy,'%')

No_files=0
for file_name in os.listdir():
  #print(file_name)
  if file_name.endswith(".txt"):
    No_files=No_files+1
    file_path = f"{file_name}"
    #print(file_path)
    with open(file_path, 'r',encoding='latin1') as curr_file:
      curr_file_read = curr_file.read()
      if Ham_OR_Spam(curr_file_read) == 'ham':
        print("0 (Ham)")
      else:
        print("1 (Spam)")

#print(No_files)
