#Paula Betina Reyes Anaya
#Ivan Tarazona Rios


# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf 
from PIL import Image
import numpy as np

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="RECONOCIMIENTO DE PRODUCTOS",
    page_icon = ":smile:",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) # Oculta el código CSS de la pantalla, ya que están incrustados en el texto de rebajas. Además, permita que Streamlit se procese de forma insegura como HTML

#st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_resource
def load_model():
    model=tf.keras.models.load_model('./producto_model.h5')
    return model
with st.spinner('Modelo está cargando..'):
    model=load_model()
    


with st.sidebar:
        st.image('productos.png')
        st.title("Reconocimiento de imagen")
        st.subheader("Reconocimiento de imagen del producto")

st.image('logo.png')
st.title("Universidad Autónoma de Bucaramanga - UNAB")
st.write("Paula Betina Reyes Anaya")
st.write("""
         El proyecto consiste que pueda reconocer los 10 productos diferentes al tomar una foto. Utiliza el reconocimiento de imagen para identificar y clasificar cada uno de los productos capturados en la fotografía.
         """
         )


def import_and_predict(image_data, model, class_names):
    
    image_data = image_data.resize((180, 180))
    
    image = tf.keras.utils.img_to_array(image_data)
    image = tf.expand_dims(image, 0) # Create a batch

    
    # Predecir con el modelo
    prediction = model.predict(image)
    index = np.argmax(prediction)
    score = tf.nn.softmax(prediction[0])
    class_name = class_names[index]
    
    return class_name, score


#class_names = open("./clases.txt", "r").readlines()
class_names = open("./clases.txt", "r", encoding="utf-8").readlines()


img_file_buffer = st.camera_input("Capture una foto para identificar un producto")    
if img_file_buffer is None:
    st.text("Por favor tome una foto")
else:
    image = Image.open(img_file_buffer)
    st.image(image, use_column_width=True)
    
    # Realizar la predicción
    class_name, score = import_and_predict(image, model, class_names)
    
    # Mostrar el resultado

    if np.max(score)>0.4:
        st.subheader(f"Tipo de producto: {class_name}")
        st.text(f"Puntuación de confianza: {100 * np.max(score):.2f}%")
    else:
        st.text(f"No se pudo determinar el tipo de producto")