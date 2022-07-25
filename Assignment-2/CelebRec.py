import streamlit as st
import boto3

st.title("Celebrity Recognition using AWS")

img_file=st.file_uploader("Upload Image",type=['png','jpg','jpeg'])

if img_file is not None:
    with open('input.jpg','wb') as f:
        f.write(img_file.getbuffer())

    client = boto3.client('rekognition')
    imageSource=open('input.jpg','rb')
    response = client.recognize_celebrities(
        Image={'Bytes':imageSource.read()}
    )
    if(len(response['CelebrityFaces'])>0):
        st.success('Celebrity Matched')
        st.markdown("Name : "+response['CelebrityFaces'][0]['Name'])
        st.markdown("URL : "+response['CelebrityFaces'][0]['Urls'][0])
    else:
        st.error("No Celebrity matched")