import streamlit as st
from englisttohindi.englisttohindi import EngtoHindi

st.title("English to Hindi Translator")


# message to be translated
message = st.text_input("Enter the text to be translated from English to Hindi")

# creating a EngtoHindi() object
res = EngtoHindi(message)
st.divider() 

# displaying the translation
if st.button("Submit"):
  st.write("The translated text is:")
  st.write(res.convert)
