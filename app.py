import pandas as pd
import numpy as np
import streamlit as st
!pip install scikit-learn

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df=pd.read_csv('mainDataset.csv',encoding= 'unicode_escape')


df.dropna(axis=0,inplace=True)
df.sort_values(by=['popularity'], ascending=False , inplace=True)
df=df.reset_index()
del df['index']

def str_to_list(s):
    l=s.split(" ")
    return l

def company(l):
    return l[0]

def kall(s):
    if s=='I':
        return s+'Kall'
    return s

def converter_str(num):
    return str(num)

def listToString(s):
     str1 = ""
     for ele in s:
        str1 += ele+" "
     return str1

def stringr(f):
    return str(f)

def price(s):
    l= s.split(",")
    return l
def og(l):
    str1=""
    str1=str1+l[0]+l[-1]
    return str1

def str_to_int(s):
    return int(s)

df['phonename']=df['phonename'].apply(str_to_list)
df['phonecompany']=df['phonename'].apply(company)
df['phonecompany']=df['phonecompany'].apply(kall)
df['phonprice']=df['phonprice'].apply(stringr).apply(price).apply(og)
df['phonprice']=df['phonprice'].apply(str_to_int)

def distribution(n):
    if n<10000:
        return 'aa'
    elif n>=10000 and n<15000:
        return 'bb'
    elif n>=15000 and n<20000:
        return 'cc'
    elif n>=20000 and n<25000:
        return 'dd'
    elif n>=25000 and n<30000:
        return 'ee'
    elif n>=30000:
        return 'ff'


df['phonprice_dist']=df['phonprice'].apply(distribution)

df['screen_size']=df['screen_size'].apply(converter_str).apply(str_to_list)
df['memory']=df['memory'].apply(converter_str).apply(str_to_list)
df['battery_size']=df['battery_size'].apply(converter_str).apply(str_to_list)
df['phonprice_dist']=df['phonprice_dist'].apply(str_to_list)
df['ram']=df['ram'].apply(converter_str).apply(str_to_list)

df['features']=df['screen_size']+df['memory']+df['battery_size']+df['ram']+df['phonprice_dist']


df['phonename']=df['phonename'].apply(listToString)

def str_to_float(s):
    return float(s)


df['features']=df['features'].apply(listToString)
df['ram']=df['ram'].apply(listToString)
df['memory']=df['memory'].apply(listToString)

df2=df[['phonename','phonecompany','phonprice','features']].copy()


#------------------------------------------------------------------------------------------

st.title('Smartphone Recommendation')
st.markdown("""
<style>
.small-font {
    font-size:13px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="small-font"> by <i>Sarthak Singh</i><br>Roll NO.-> <i>21238</i><br>Branch -> <i>Electronics and Communication Engineering</i></p>', unsafe_allow_html=True)


ram = st.selectbox(
    'Choose the RAM in GigaBytes',
    (' ','1','2','3','4','6','8','12','16')
)
display='5.5'
battery='5000'
memory = st.text_area("Input the memory requirement in GigaBytes")
screen = st.text_area("Input the display requirement in Inches")
battery = st.text_area("Input the battery requirement in mAh")
price = st.selectbox(
    'Choose the Price range',
    ('Below 10,000','10,000 - 15,000','15,000 - 20,000','20,000 - 25,000','25,000 - 30,000','Above 30,000')
)

if price=='Below 10,000':
    price='aa'
elif price=='10,000 - 15,000':
    price='bb'
elif price == '15,000 - 20,000':
    price = 'cc'
elif price == '20,000 - 25,000':
    price='dd'
elif price=='25,000 - 30,000':
    price='ee'
elif price=='Above 30,000':
    price='ff'


#------------------------------------------------------------------------------

str_new=screen+" "+memory+" "+battery+" "+ram+" "+price

new_row={"phonename":"xyz","phonecompany":"Company","phonprice":price,"features":str_new}
df2 = df2.append(new_row, ignore_index=True)

cv=CountVectorizer(max_features=5000,stop_words="english")
vectors=cv.fit_transform( df2['features']).toarray()

similarity=cosine_similarity(vectors)

def recommend(phone):
    print("COMPANY","----","MODEL","----","PRICE")
    phone_index=df2[df2['phonename']== phone].index[0]
    distances=similarity[phone_index]
    phones_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:200]
    l=[]
    l1=[]
    for i in phones_list:
        if(df2.iloc[i[0]].phonename!="xyz"):
            tups=('Company : '+df2.iloc[i[0]].phonecompany,'Model : '+df2.iloc[i[0]].phonename,'Price : '+str(df2.iloc[i[0]].phonprice))
            l.append(tups)
    return l

def image_abc(phone):
    phone_index=df2[df2['phonename']== phone].index[0]
    distances=similarity[phone_index]
    phones_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:200]
    l=[]
    for i in phones_list:
        if(df2.iloc[i[0]].phonename!="xyz"):
            l.append(df.iloc[i[0]].image)
    return l


st.markdown("""
<style>
.big-font {
    font-size:23px !important;
}
</style>
""", unsafe_allow_html=True)


if st.button('Recommend'):
    count=1
    counter=5
    new = recommend('xyz')
    images=image_abc('xyz')
    st.markdown('<p class="big-font">Here are the  <i>RECOMMENDATIONS</i>  for you ! &#128241</p>', unsafe_allow_html=True)
    st.write("--------------------------------------------------------------------------")
    for i in range(0,len(new)):
        if counter!=0:
            col1, mid, col2 = st.columns([50,4,20])
            with col1:
                st.write(str(count),')')
                for j in new[i]:
                    st.write(j)
            with col2:
                st.image(str(images[i]), width = 100)
            st.write("--------------------------------------------------------------------------")
            count=count+1
            counter-=1
