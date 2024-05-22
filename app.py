from flask import Flask, request
import os

import numpy as np
import pandas as pd
import cv2
import pytesseract
import re
import string
import spacy

app= Flask(__name__)

@app.post("/prueba")
def prueba():
    file =request.files['file']
    filename=file.filename
    print('Image saved in = ',filename)
    name,ext=filename.split('.')
    save_filename='upload.'+ext
    BASE_DIR = os.getcwd()
    SAVE_DIR = BASE_DIR + '/media'
    upload_image_path=os.path.join(SAVE_DIR,save_filename)
    file.save(upload_image_path)
    print('Image saved in = ',upload_image_path)

    ##reconocimiento de caracteres

    tessData = pytesseract.image_to_data(upload_image_path)

    # convert into dataframe
    tessList = list(map(lambda x:x.split('\t'), tessData.split('\n')))
    df = pd.DataFrame(tessList[1:],columns=tessList[0])
    df.dropna(inplace=True) # drop missing values
    df['text'] = df['text'].apply(cleanText)

    # convet data into content
    df_clean = df.query('text != "" ')
    content = " ".join([w for w in df_clean['text']])
    print(content)
    # get prediction from NER model
    doc = model_ner(content)

    # converting doc in json
    docjson = doc.to_json()
    doc_text = docjson['text']

    # creating tokens
    datafram_tokens = pd.DataFrame(docjson['tokens'])
    datafram_tokens['token'] = datafram_tokens[['start','end']].apply(
        lambda x:doc_text[x[0]:x[1]] , axis = 1)
    
    right_table = pd.DataFrame(docjson['ents'])[['start','label']]
    datafram_tokens = pd.merge(datafram_tokens,right_table,how='left',on='start')
    datafram_tokens.fillna('O',inplace=True)

        # join lable to df_clean dataframe
    df_clean['end'] = df_clean['text'].apply(lambda x: len(x)+1).cumsum() - 1 
    df_clean['start'] = df_clean[['text','end']].apply(lambda x: x[1] - len(x[0]),axis=1)

    # inner join with start 
    dataframe_info = pd.merge(df_clean,datafram_tokens[['start','token','label']],how='inner',on='start')

        # Bounding Box

    bb_df = dataframe_info.query("label != 'O' ")

    grp_gen = groupgen()


    bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])
    bb_df['group'] = bb_df['label'].apply(grp_gen.getgroup)

    # right and bottom of bounding box
    bb_df[['left','top','width','height']] = bb_df[['left','top','width','height']].astype(int)
    bb_df['right'] = bb_df['left'] + bb_df['width']
    bb_df['bottom'] = bb_df['top'] + bb_df['height']

    # tagging: groupby group
    col_group = ['left','top','right','bottom','label','token','group']
    group_tag_img = bb_df[col_group].groupby(by='group')
    img_tagging = group_tag_img.agg({

        'left':min,
        'right':max,
        'top':min,
        'bottom':max,
        'label':np.unique,
        'token':lambda x: " ".join(x)

    })
    image2 = cv2.imread(upload_image_path)
    img_bb = image2.copy()
    #recognize without img_bb

    for l,r,t,b,label,token in img_tagging.values:
        cv2.rectangle(img_bb,(l,t),(r,b),(0,255,0),2)
        ##cv.putText(img_bb,label,(l,t), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255),2)
        label = str(label)
        cv2.putText(img_bb,label,(l,t),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)


    # Entities

    info_array = dataframe_info[['token','label']].values
    entities = dict(LUGARFECHA=[],RECEPTOR=[],EMISOR=[],CODIGO=[],CONTENIDO=[],FIRMA=[])
    previous = 'O'

    for token, label in info_array:
        bio_tag = label[0]
        label_tag = label[2:]
        
        # step -1 parse the token
        text = parser(token,label_tag)
        
        if bio_tag in ('B','I'):
            
            if previous != label_tag:
                entities[label_tag].append(text)
                
            else:
                if bio_tag == "B":
                    entities[label_tag].append(text)
                    
                else:
                    if label_tag in ('CODIGO','CONTENIDO'):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text
                        
                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text
                        
        
        
        previous = label_tag

    return entities
model_ner = spacy.load('./output/model-best/')

def cleanText(txt):
    whitespace = string.whitespace
    punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    tableWhitespace = str.maketrans('','',whitespace)
    tablePunctuation = str.maketrans('','',punctuation)
    text = str(txt)
    #text = text.lower()
    removewhitespace = text.translate(tableWhitespace)
    removepunctuation = removewhitespace.translate(tablePunctuation)
    
    return str(removepunctuation)

def parser(text,label):            
    if label == 'LUGARFECHA':
        ##text = text.lower()
        allow_special_char = '-,.'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char),'',text)
         
    elif label == 'RECEPTOR':
        allow_special_char = ',-:.'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char),'',text)
    
    elif label == 'EMISOR':
        allow_special_char = ',-:.'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char),'',text)
        
    elif label == 'CODIGO':
        ##text = text.lower()
        allow_special_char = '-,º./'
        text = re.sub(r'[^A-Z0-9{} ]'.format(allow_special_char),'',text)
        
    elif label == 'CONTENIDO':
        ##text = text.lower()
        allow_special_char = '-@:;/.%#\º""'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char),'',text)
       
    return text
# group the label
class groupgen():
    def __init__(self):
        self.id = 0
        self.text = ''
        
    def getgroup(self,text):
        if self.text == text:
            return self.id
        else:
            self.id +=1
            self.text = text
            return self.id
        
