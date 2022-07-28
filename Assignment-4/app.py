import streamlit as st
import boto3

st.title("Listing objects in a bucket")

client = boto3.client('s3')

files = st.file_uploader("Upload Files",type=['txt','py','png','jpg','docx','pdf'],accept_multiple_files=True)
bucket_name=st.text_input("Bucket Name :")
if len(files)>0 and st.button('upload'):
    file_details={}
    for i in range(len(files)):
        file_details['name']=files[i].name
        file_details['type']=files[i].type
        file_details['size']=files[i].size
        print(file_details)
        client.upload_file(file_details['name'],bucket_name,file_details['name'])
        st.success("file uploaded")
if st.button('List files'):
    s3=boto3.resource('s3')
    my_bucket = s3.Bucket(bucket_name)
    objects=list(my_bucket.objects.all())
    st.success("The no.of object in the %s is %d." %(my_bucket.name,len(objects)))
    for object in objects:
        result = f"bucket : {my_bucket.name}, file name : {object.key}"
        st.markdown(result)