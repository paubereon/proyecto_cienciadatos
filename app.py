#Paula Betina Reyes Anaya
#Ivan Tarazona Rios

#Importamos la libreria para los modelo
import streamlit as st  
import tensorflow as tf 
from PIL import Image
import numpy as np
from gtts import gTTS
import os
import base64
import io

#Configuramos la página de Streamlit
import streamlit as st

st.set_page_config(page_title="Aplicación de Inventario Inteligente con Sugerencias Dinámicas de Recetas", 
                   page_icon="https://images.freeimages.com/fic/images/icons/61/dragon_soft/512/usuario.png",
                   layout="centered",
                   initial_sidebar_state="auto")


# Ocultar elementos de Streamlit
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Cargar el modelo de detección de frutas
@st.cache_resource
def load_detection_model():
    return tf.keras.models.load_model('./fruit_detection_model.h5')


# Interfaz de usuario
with st.sidebar:
    st.image('productos.png')
    st.title("Reconocimiento de Productos")
    st.subheader("Capture una imagen para identificar los productos")

st.image('logo.png')
st.markdown('<h3 style="font-size: 18px;">Elaborado por: Paula Betina Reyes Anaya y Ivan Tarazona Rios</h3>', unsafe_allow_html=True)

# Definimos el título y la descripción de la aplicación
st.title("Universidad Autónoma de Bucaramanga - UNAB")
st.header('Aplicación de Sugerencia de Recetas Basada en Inventario', divider='rainbow')
st.subheader('Creación de un Inventario Digital de Ingredientes')

with st.container(border=True):
    st.subheader("Aplicación para Identificar Ingredientes y Sugerir Recetas")
    # wave es un emoji
    st.write("Realizado por Paula Betina Reyes y Ivan Tarazona Rios :wave:")
    st.write("""
**OBJETIVO**:
El objetivo de este proyecto es permitir que el usuario capture o suba una foto de algun ingrediente. La aplicación identifica los ingredientes mediante visión por computadora y crea un inventario digital. Además, el usuario tiene la opción de confirmar o editar los ingredientes detectados. Con este inventario, el sistema sugiere tres recetas adaptadas a lo que el usuario realmente tiene en su hogar. Al seleccionar una receta, se muestra el detalle y se reproduce mediante text-to-speech, ofreciendo una experiencia interactiva y personalizada.
    """)

  
with st.container( border=True):
  st.subheader("PROYECTO GOOGLE COLAB")
  st.write(""" Enlace de los modelos de Redes Neuronales entrenados en Google Colab con las librerias
        https://colab.research.google.com/drive/1W3ROKwPQ9fVJZh2noxf-K3a19SwMkieD?usp=sharing""")

# Poner un título más grande
st.markdown("<h2 style='text-align: center;'>Sube una foto de un ingrediente</h2>", unsafe_allow_html=True)

# Subir la imagen de la alacena o nevera
uploaded_file = st.file_uploader("Sube una foto de un ingrediente", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Foto del ingrediente", use_column_width=True)

#al ssbir la foro me tiene que decir que es y como es la predicion
# Cargar nombres de productos
class_names = [line.strip() for line in open("./frutas.txt", "r", encoding="utf-8").readlines()]

# Función para realizar la predicción
def import_and_predict(image_data, model, class_names):
    image_data = image_data.resize((180, 180))
    image = tf.keras.utils.img_to_array(image_data)
    image = tf.expand_dims(image, 0)  # Crear batch
    prediction = model.predict(image)
    index = np.argmax(prediction)
    score = tf.nn.softmax(prediction[0])
    class_name = class_names[index]
    return class_name, score


import streamlit as st

# Cargar ingredientes desde el archivo
with open("frutas.txt", "r", encoding="utf-8") as file:
    ingredientes_disponibles = [line.strip() for line in file.readlines()]

# Diccionario de recetas con descripción y pasos
todas_las_recetas = {
"Tarta de manzana": {
    "imagen": "https://th.bing.com/th/id/OIP.DlUn8ytvF4Yq5P4RBdlyvwHaFj?rs=1&pid=ImgDetMain",
    "descripcion": "¡Qué rica es esta receta de tarta de manzana! Con un interior suave y muy jugoso, esta tarta es toda una delicia. Tiene, además, la ventaja de que es un postre muy fácil de preparar, prácticamente solo hay que batir ingredientes y dejar que el horno trabaje para poder disfrutarla.",
    "ingredientes": ["Harina", "Azúcar", "Canela", "Huevos", "Mantequilla"],
    "pasos": [
        "Paso 1: Preparar la masa. Primero, en un bol grande, ralla 3 manzanas (deja una manzana sin rallar para después). Añade 2 huevos, 100g de azúcar, 1 cucharadita de canela en polvo y 100ml de leche. Bate todo con unas varillas hasta que obtengas una mezcla homogénea y suave.",
        "Paso 2: Agregar las harinas. Tamiza 200g de harina y 1 cucharadita de levadura en polvo directamente sobre la mezcla anterior. Revuelve con una espátula hasta que se integren bien, pero sin que queden grumos. El objetivo es que la masa quede suave y espesa.",
        "Paso 3: Preparar la base. En un molde para tartas, unta un poco de mantequilla (unos 30g) en el fondo y espolvorea con harina. A continuación, vierte la masa que has preparado y extiende con una espátula para que quede uniforme.",
        "Paso 4: Preparar las manzanas. Pela la manzana que habías reservado, córtala en finas láminas y colócala sobre la masa en forma de espiral, cubriendo toda la tarta.",
        "Paso 5: Hornear. Precalienta el horno a 180°C y coloca la tarta en el centro. Hornea durante unos 40 minutos, o hasta que la superficie esté dorada y al pincharla con un palillo salga limpio.",
        "Paso 6: Disfrutar. Deja enfriar un poco antes de servir, y acompáñala con un poco de crema o helado de vainilla. ¡Te encantará!"
    ]
},
    "Compota de manzana": {
    "imagen": "https://th.bing.com/th/id/OIP.h0oJQUpEM-3Hhh-d0VPvLQHaE4?rs=1&pid=ImgDetMain",
    "descripcion": "La compota de manzana es un postre suave y dulce, ideal para los más pequeños de la casa o como acompañamiento de otros postres. ¡Es como un abrazo cálido en cada cucharada!",
    "ingredientes": ["Azúcar", "Canela"],
    "pasos": [
        "Paso 1: Pelar y cortar las manzanas. Pela y quita el corazón de 4 manzanas. Corta las manzanas en trozos pequeños.",
        "Paso 2: Cocinar las manzanas. Coloca las manzanas en una cacerola grande junto con 100ml de agua, 2 cucharadas de azúcar y 1 cucharadita de canela en polvo.",
        "Paso 3: Cocinar a fuego lento. Cocina a fuego medio durante unos 15-20 minutos, removiendo ocasionalmente. Las manzanas deben deshacerse y formar una mezcla espesa.",
        "Paso 4: Añadir el toque de limón. Exprime el jugo de medio limón sobre la compota y mezcla bien.",
        "Paso 5: Ajustar el dulzor. Si te gusta más dulce, agrega un poco más de azúcar al gusto y cocina unos minutos más.",
        "Paso 6: Servir. Sirve la compota tibia o fría, sola o acompañada de un poco de crema o yogur natural. ¡Es deliciosa y reconfortante!"
    ]
},
    "Tarta Tatín de manzana": {
    "imagen": "https://imag.bonviveur.com/tarta-de-manzana.jpg",
    "descripcion": "La Tarta Tatín es una deliciosa tarta invertida, donde las manzanas caramelizadas son el centro de atención. Su combinación de dulce y ácido hace de esta receta un postre espectacular.",
    "ingredientes": ["Mantequilla", "Azúcar", "Masa quebrada", "Canela"],
    "pasos": [
        "Paso 1: Preparar las manzanas. Pela 4 manzanas, quita el corazón y córtalas en cuartos. Reserva.",
        "Paso 2: Caramelizar el azúcar. En una sartén grande, derrite 100g de mantequilla a fuego medio. Añade 150g de azúcar y deja que se derrita y se forme un caramelo dorado.",
        "Paso 3: Colocar las manzanas. Cuando el caramelo esté listo, coloca las manzanas sobre él de forma ordenada, cubriendo toda la sartén. Cocina las manzanas durante 10 minutos, dándoles vuelta para que se caramelicen por igual.",
        "Paso 4: Añadir la canela. Espolvorea una pizca de canela sobre las manzanas para darle un toque extra de sabor.",
        "Paso 5: Preparar la base. Cubre las manzanas con una capa de masa quebrada (puedes comprarla ya lista o hacerla casera). Asegúrate de que la masa cubra completamente las manzanas.",
        "Paso 6: Hornear. Precalienta el horno a 180°C y hornea la tarta durante 30 minutos, o hasta que la masa esté dorada y crujiente.",
        "Paso 7: Voltear la tarta. Una vez que la tarta esté lista, retírala del horno y deja reposar unos minutos. Luego, voltea la sartén sobre un plato grande para desmoldar la tarta con cuidado.",
        "Paso 8: Disfrutar. Sirve esta deliciosa tarta Tatín de manzana tibia, acompañada de crema fresca o helado de vainilla. ¡Un postre espectacular!"
    ]
},
    "Mermelada de albaricoque": {
    "imagen": "https://th.bing.com/th/id/OIP.p0IKfCrJsGPpEnUUODPPIwHaE8?rs=1&pid=ImgDetMain",
    "descripcion": "Una mermelada dulce y sabrosa, ideal para untar sobre pan o para rellenar pasteles y tartas. Con el sabor ácido del albaricoque, esta receta es un placer.",
    "ingredientes": ["Azúcar", "Agua"],
    "pasos": [
        "Paso 1: Preparar los albaricoques. Lava y pela 1 kg de albaricoques. Corta los albaricoques por la mitad, quita el hueso y córtalos en trozos pequeños para facilitar la cocción.",
        "Paso 2: Cocinar la fruta. Coloca los albaricoques en una cacerola grande y añade 750g de azúcar. Agrega el jugo de medio limón (aproximadamente 2 cucharadas). Remueve bien para que todo quede cubierto con el azúcar.",
        "Paso 3: Cocción a fuego lento. Cocina a fuego medio, removiendo con una cuchara de madera cada 10 minutos para evitar que se pegue al fondo. La mezcla debe cocerse durante unos 30-40 minutos, hasta que la fruta se haya deshecho y la mezcla tenga una textura espesa.",
        "Paso 4: Prueba de consistencia. Para saber si la mermelada está lista, coloca un poco de la mezcla en un plato frío. Si al inclinarlo no se escurre, está lista. Si no, sigue cocinando un poco más.",
        "Paso 5: Envasado. Una vez que la mermelada esté lista, retira del fuego y, con cuidado, vierte en frascos de vidrio previamente esterilizados. Cierra los frascos y voltea para que el vacío se haga. Deja enfriar completamente.",
        "Paso 6: Disfrutar. Ya tienes una deliciosa mermelada de albaricoque para acompañar tu pan tostado o para usar en tus postres favoritos."
    ]
},
    "Tarta de albaricoque": {
    "imagen": "https://th.bing.com/th/id/OIP.HGiSNOXxGUf2L2Tgani3qgHaJ4?w=585&h=780&rs=1&pid=ImgDetMain",
    "descripcion": "Esta tarta de albaricoque es una delicia fresca y afrutada, con una base crujiente y una capa de albaricoques caramelizados que se derriten en la boca. ¡Es perfecta para el verano!",
    "ingredientes": ["Huevo", "Harina", "Azúcar", "Mantequilla"],
    "pasos": [
        "Paso 1: Preparar los albaricoques. Lava 8 albaricoques y córtalos por la mitad, quitándoles el hueso. Corta cada mitad en trozos más pequeños.",
        "Paso 2: Hacer la base. En un bol grande, mezcla 200g de harina con 100g de mantequilla fría cortada en trozos. Añade 50g de azúcar y 1 huevo. Mezcla hasta formar una masa homogénea.",
        "Paso 3: Enfriar la masa. Envuelve la masa en plástico transparente y refrigérala durante 30 minutos.",
        "Paso 4: Preparar la base de la tarta. Estira la masa en una superficie enharinada y colócala en un molde para tarta previamente enharinado. Pincha la base con un tenedor y hornea a 180°C durante 15 minutos.",
        "Paso 5: Caramelizar los albaricoques. Mientras la base se hornea, coloca los trozos de albaricoque en una sartén con 2 cucharadas de azúcar y cocina a fuego medio, removiendo de vez en cuando, hasta que los albaricoques se caramelicen, unos 10 minutos.",
        "Paso 6: Montar la tarta. Una vez que la base esté lista, saca del horno y coloca los albaricoques caramelizados sobre ella.",
        "Paso 7: Hornear de nuevo. Hornea la tarta durante 20-25 minutos a 180°C o hasta que los bordes de la base estén dorados.",
        "Paso 8: Servir. Deja enfriar la tarta antes de servirla. Puedes decorarla con un poco de azúcar glas por encima o acompañarla con una bola de helado de vainilla."
    ]
},
    "Ensalada de albaricoque y queso de cabra": {
    "imagen": "https://th.bing.com/th/id/OIP.LfrNv5shYrQxJZtD89hivgHaEK?rs=1&pid=ImgDetMain",
    "descripcion": "Esta ensalada fresca y sabrosa combina la dulzura de los albaricoques con el sabor cremoso y ligeramente salado del queso de cabra. ¡Perfecta como entrada o acompañamiento para una comida ligera!",
    "ingredientes": ["Queso de cabra", "Espinacas", "Almendras", "Miel", "Aceite de oliva"],
    "pasos": [
        "Paso 1: Preparar los albaricoques. Lava 6 albaricoques, córtalos por la mitad y quítales el hueso. Luego, corta cada mitad en rodajas finas.",
        "Paso 2: Preparar las espinacas. Lava bien un puñado de espinacas frescas y colócalas en un bol grande.",
        "Paso 3: Agregar el queso de cabra. Corta 100g de queso de cabra en trozos pequeños y agrégalo al bol con las espinacas.",
        "Paso 4: Tostar las almendras. En una sartén pequeña, tuesta ligeramente un puñado de almendras (aproximadamente 30g) a fuego medio durante 3-4 minutos, removiendo para que no se quemen. Luego, pica las almendras en trozos pequeños.",
        "Paso 5: Montar la ensalada. Añade las rodajas de albaricoque a la ensalada junto con las almendras tostadas.",
        "Paso 6: Preparar el aderezo. En un tazón pequeño, mezcla 2 cucharadas de aceite de oliva con 1 cucharada de miel y una pizca de sal y pimienta al gusto. Bate bien hasta que todos los ingredientes estén bien integrados.",
        "Paso 7: Aliñar la ensalada. Vierte el aderezo sobre la ensalada y mezcla suavemente para que todos los ingredientes se impregnen bien.",
        "Paso 8: Servir. Sirve la ensalada de albaricoque y queso de cabra de inmediato como entrada o acompañamiento. ¡Disfruta de esta ensalada fresca, dulce y salada!"
    ]
},
   "Guacamole": {
    "imagen": "https://th.bing.com/th/id/R.37887bae4f040cd0fb2f15a22b2b7849?rik=%2bBbHjFpNPO1CJQ&pid=ImgRaw&r=0",
    "descripcion": "El guacamole es una receta fresca, ideal como aperitivo o acompañamiento. Con aguacate cremoso y especias, se convierte en el dip perfecto para nachos o tacos.",
    "ingredientes": ["Cebolla", "Jitomate", "Cilantro"],
    "pasos": [
        "Paso 1: Preparar los aguacates. Abre 2 aguacates por la mitad, quita el hueso y extrae la pulpa con una cuchara. Colócala en un tazón grande.",
        "Paso 2: Machacar. Con un tenedor, comienza a machacar el aguacate hasta que quede una mezcla suave pero con algunos trozos pequeños de aguacate, lo que le dará textura.",
        "Paso 3: Agregar la cebolla. Pica finamente ¼ de cebolla morada y añádela al aguacate machacado.",
        "Paso 4: Añadir el tomate y el cilantro. Corta un jitomate en cubos pequeños y agrégalo al tazón. Pica 2 cucharadas de cilantro fresco y añádelas a la mezcla.",
        "Paso 5: Sazonar con limón y sal. Exprime el jugo de 1 limón sobre la mezcla y sazona con sal y pimienta al gusto. Remueve todo para que los sabores se integren bien.",
        "Paso 6: Disfrutar. Sirve el guacamole con nachos, tortillas o como acompañamiento de tus tacos. ¡Lo disfrutarás!"
    ]
},

"Tostadas de aguacate con huevo poché": {
    "imagen": "https://mandolina.co/wp-content/uploads/2024/05/tostada-aguacate-1.jpg",
    "descripcion": "Estas tostadas de aguacate con huevo poché son perfectas para un desayuno saludable y delicioso. El aguacate cremoso combina a la perfección con el huevo suave y el toque crujiente de la tostada.",
    "ingredientes": ["Pan integral", "Huevo", "Aceite de oliva", "Sal", "Pimienta", "Pimentón"],
    "pasos": [
        "Paso 1: Preparar el aguacate. Pela y deshuesa 1 aguacate maduro. Tritura la pulpa con un tenedor y añádele el jugo de medio limón, sal y pimienta al gusto.",
        "Paso 2: Tostar el pan. Tuesta dos rebanadas de pan integral hasta que estén doradas y crujientes.",
        "Paso 3: Cocinar el huevo poché. En una cacerola con agua caliente (sin que llegue a hervir), agrega un chorro de vinagre. Rompe un huevo y, con cuidado, viértelo en el agua caliente. Cocina durante 3-4 minutos hasta que la clara esté firme pero la yema aún esté líquida.",
        "Paso 4: Montar las tostadas. Unta una capa generosa de aguacate triturado sobre las tostadas de pan integral.",
        "Paso 5: Colocar el huevo poché. Con cuidado, coloca un huevo poché sobre cada tostada de aguacate.",
        "Paso 6: Añadir el toque final. Rocía las tostadas con un poco de aceite de oliva y espolvorea pimentón y una pizca de sal por encima.",
        "Paso 7: Servir. Sirve estas deliciosas tostadas de aguacate con huevo poché de inmediato. ¡Son el desayuno perfecto para comenzar el día!"
    ]
},
    "Ensalada de aguacate y tomate con vinagreta de mostaza": {
    "imagen": "https://www.madamecuisine.de/wp-content/uploads/2022/08/caprese-bordelais02-1365x2048.jpg",
    "descripcion": "Esta ensalada fresca y colorida es perfecta como acompañante para carnes o como plato principal para una comida ligera. El aguacate cremoso y el tomate jugoso se combinan maravillosamente con la vinagreta de mostaza.",
    "ingredientes": ["Cebolla roja", "Aceite de oliva", "Mostaza", "Vinagre balsámico", "Miel", "Sal", "Pimienta"],
    "pasos": [
        "Paso 1: Preparar los ingredientes. Lava 2 tomates y córtalos en cubos. Pela y corta en rodajas finas 1 cebolla roja pequeña. Pela y deshuesa 1 aguacate, luego córtalo en cubos.",
        "Paso 2: Preparar la vinagreta. En un tazón pequeño, mezcla 2 cucharadas de mostaza, 1 cucharada de vinagre balsámico, 2 cucharadas de miel, 3 cucharadas de aceite de oliva, sal y pimienta al gusto. Bate bien hasta que la vinagreta esté bien emulsionada.",
        "Paso 3: Mezclar la ensalada. En un bol grande, agrega los tomates, la cebolla y el aguacate. Vierte la vinagreta de mostaza sobre los ingredientes y mezcla suavemente para que se impregnen bien.",
        "Paso 4: Servir. Sirve la ensalada de aguacate y tomate inmediatamente, como acompañamiento o plato principal. ¡Es fresca, ligera y llena de sabor!"
    ]
},
    "Pan de plátano": {
    "imagen": "https://th.bing.com/th/id/OIP.a3tuAbaCyj3pZ4xjWLpP-wHaFT?rs=1&pid=ImgDetMain",
    "descripcion": "Un pan suave y esponjoso con el dulce sabor del plátano maduro. Ideal para acompañar un café o disfrutar a cualquier hora del día.",
    "ingredientes": ["Harina", "Azúcar", "Huevos", "Mantequilla", "Bicarbonato", "Sal", "Vainilla"],
    "pasos": [
        "Paso 1: Preparar los plátanos. Pela 3 plátanos maduros y aplástalos con un tenedor hasta obtener un puré suave.",
        "Paso 2: Batir los ingredientes. En un bol, bate 100g de mantequilla derretida con 200g de azúcar. Agrega 2 huevos y la esencia de 1 cucharadita de vainilla. Incorpora el puré de plátano.",
        "Paso 3: Mezclar los secos. En otro bol, mezcla 250g de harina, 1 cucharadita de bicarbonato y una pizca de sal.",
        "Paso 4: Combinar. Agrega los ingredientes secos a los ingredientes húmedos y mezcla hasta obtener una masa suave.",
        "Paso 5: Hornear. Vierte la masa en un molde engrasado y hornea a 180°C durante 60 minutos o hasta que al insertar un palillo, éste salga limpio.",
        "Paso 6: Servir. Deja enfriar y disfruta de este delicioso pan de plátano."
    ]
},
    "Smoothie de plátano": {
    "imagen": "https://th.bing.com/th/id/OIP.xf4R_aexbz_wHFtDWSh2jQHaE7?rs=1&pid=ImgDetMain",
    "descripcion": "Un smoothie cremoso y refrescante, perfecto para empezar el día con energía o como un snack saludable.",
    "ingredientes": ["Leche", "Miel", "Hielo"],
    "pasos": [
        "Paso 1: Preparar los ingredientes. Pela 1 plátano maduro y córtalo en trozos.",
        "Paso 2: Licuar. Coloca el plátano en una licuadora junto con 200 ml de leche (puedes usar leche de almendras o de coco si prefieres) y 1 cucharadita de miel.",
        "Paso 3: Agregar hielo. Añade un puñado de hielo y licúa hasta obtener una mezcla suave y cremosa.",
        "Paso 4: Servir. Vierte el smoothie en un vaso y disfrútalo al momento, ¡es perfecto para refrescarte y disfrutar del sabor del plátano!"
    ]
},
    "Tarta de plátano": {
    "imagen": "https://th.bing.com/th/id/OIP.U734zWNRNakReoR8FdnmQQHaE8?w=750&h=500&rs=1&pid=ImgDetMain",
    "descripcion": "Una tarta suave con un toque dulce y natural del plátano. Perfecta para los amantes de los postres sencillos y deliciosos.",
    "ingredientes": ["Harina", "Azúcar", "Huevos", "Leche", "Mantequilla", "Polvo de hornear"],
    "pasos": [
        "Paso 1: Preparar los plátanos. Pela 2 plátanos maduros y hazlos puré con un tenedor.",
        "Paso 2: Batir los ingredientes. En un bol grande, bate 100g de mantequilla con 150g de azúcar. Agrega 2 huevos y sigue batiendo.",
        "Paso 3: Añadir los plátanos. Incorpora el puré de plátano a la mezcla de mantequilla y azúcar.",
        "Paso 4: Mezclar los secos. En otro bol, tamiza 200g de harina con 1 cucharadita de polvo de hornear.",
        "Paso 5: Hornear. Vierte la mezcla en un molde engrasado y hornea a 180°C durante 35-40 minutos o hasta que al insertar un palillo, éste salga limpio.",
        "Paso 6: Servir. Deja enfriar antes de servir y disfruta de esta deliciosa tarta de plátano."
    ]
},
    "Mermelada de mora": {
    "imagen": "https://th.bing.com/th/id/OIP.0bqj4B85yL0d1Wsf5uHDSQHaFU?rs=1&pid=ImgDetMain",
    "descripcion": "Una mermelada casera y deliciosa, perfecta para acompañar tostadas o postres. El sabor de la mora le da un toque único.",
    "ingredientes": ["Azúcar", "Jugo de limón", "Pectina"],
    "pasos": [
        "Paso 1: Preparar las moras. Lava 500g de moras frescas y colócalas en una cacerola.",
        "Paso 2: Cocinar las moras. Agrega 250g de azúcar y el jugo de 1 limón. Cocina a fuego medio, removiendo constantemente.",
        "Paso 3: Añadir pectina. Si prefieres una mermelada más espesa, agrega 1 cucharada de pectina y cocina durante 10 minutos más.",
        "Paso 4: Envasar. Vierte la mermelada caliente en frascos esterilizados, cierra bien y deja enfriar.",
        "Paso 5: Servir. Disfruta de esta mermelada de mora sobre pan, galletas o yogur."
    ]
},
    "Tarta de mora": {
    "imagen": "https://th.bing.com/th/id/OIP.BiwSqQCJ4NMwpHqGNkWgggHaEK?rs=1&pid=ImgDetMain",
    "descripcion": "Una tarta ligera con un toque ácido y dulce de las moras. ¡El postre perfecto para cualquier ocasión!",
    "ingredientes": ["Harina", "Azúcar", "Mantequilla", "Huevos", "Levadura"],
    "pasos": [
        "Paso 1: Preparar las moras. Lava 300g de moras frescas y reserva.",
        "Paso 2: Preparar la masa. En un bol, mezcla 200g de harina con 100g de azúcar y 1 cucharadita de levadura. Agrega 100g de mantequilla derretida y 2 huevos.",
        "Paso 3: Hornear la base. Vierte la mezcla en un molde engrasado y hornea a 180°C durante 25 minutos.",
        "Paso 4: Añadir las moras. Después de que la base se haya enfriado un poco, coloca las moras sobre la tarta.",
        "Paso 5: Hornear de nuevo. Vuelve a hornear durante 10 minutos a 180°C.",
        "Paso 6: Servir. Deja enfriar completamente antes de servir y disfruta de esta deliciosa tarta."
    ]
},
    "Smoothie de mora": {
    "imagen": "https://www.recetasfacilesreunidas.com/files/styles/receta_full/public/receta/smoothie-de-mora.jpg",
    "descripcion": "Un smoothie refrescante y lleno de antioxidantes, ideal para una bebida saludable en cualquier momento del día.",
    "ingredientes": ["Leche", "Miel", "Hielo"],
    "pasos": [
        "Paso 1: Preparar las moras. Lava 200g de moras frescas.",
        "Paso 2: Licuar. Coloca las moras en una licuadora junto con 200ml de leche (puedes usar leche de almendras) y 1 cucharada de miel.",
        "Paso 3: Agregar hielo. Añade un puñado de hielo y licúa hasta obtener una mezcla suave.",
        "Paso 4: Servir. Sirve el smoothie de mora inmediatamente en un vaso frío."
    ]
},
    "Mermelada de arándano": {
    "imagen": "https://i0.wp.com/citymagazine.si/wp-content/uploads/2016/06/shutterstock_336914663.jpg",
    "descripcion": "Una mermelada deliciosa que captura todo el sabor de los arándanos frescos. Perfecta para acompañar tostadas o postres.",
    "ingredientes": [ "Azúcar", "Pectina"],
    "pasos": [
        "Paso 1: Preparar los arándanos. Lava 500g de arándanos y colócalos en una cacerola.",
        "Paso 2: Cocinar. Agrega 250g de azúcar y el jugo de 1 limón. Cocina a fuego medio hasta que los arándanos se deshagan.",
        "Paso 3: Añadir pectina. Si quieres una mermelada más espesa, agrega 1 cucharadita de pectina y cocina 10 minutos más.",
        "Paso 4: Envasar. Vierte la mermelada en frascos esterilizados, cierra bien y deja enfriar.",
        "Paso 5: Servir. Disfruta de la mermelada de arándano sobre pan o en yogur."
    ]
},
    "Tarta de arándano": {
    "imagen": "https://recetastips.com/wp-content/uploads/2023/07/Tarta-de-Arandanos.jpg",
    "descripcion": "Una tarta fresca y deliciosa, con la acidez de los arándanos que se complementa perfectamente con la dulzura de la masa.",
    "ingredientes": ["Harina", "Azúcar", "Mantequilla", "Leche", "Huevos"],
    "pasos": [
        "Paso 1: Preparar los arándanos. Lava 200g de arándanos y resérvalos.",
        "Paso 2: Preparar la masa. Mezcla 250g de harina con 150g de azúcar y 100g de mantequilla derretida. Agrega 1 huevo y 2 cucharadas de leche. Mezcla hasta formar una masa.",
        "Paso 3: Hornear la base. Vierte la masa en un molde engrasado y hornea a 180°C durante 20 minutos.",
        "Paso 4: Añadir los arándanos. Coloca los arándanos sobre la base horneada y hornea durante 15 minutos más.",
        "Paso 5: Servir. Deja enfriar antes de servir esta deliciosa tarta."
    ]
},
    "Smoothie de arándano": {
    "imagen": "https://th.bing.com/th/id/OIP.MCi97CGxb51zZL3lNGeD2AHaE6?rs=1&pid=ImgDetMain",
    "descripcion": "Este smoothie es refrescante, lleno de antioxidantes y muy fácil de hacer. ¡Ideal para un snack saludable!",
    "ingredientes": ["Yogur", "Miel", "Hielo"],
    "pasos": [
        "Paso 1: Preparar los arándanos. Lava 150g de arándanos frescos.",
        "Paso 2: Licuar. Coloca los arándanos en la licuadora con 200g de yogur natural y 1 cucharadita de miel.",
        "Paso 3: Agregar hielo. Añade un puñado de hielo y licúa hasta obtener una mezcla suave.",
        "Paso 4: Servir. Sirve el smoothie de arándano en un vaso frío y disfruta."
    ]
},
    "Mermelada de cereza": {
    "imagen": "https://th.bing.com/th/id/OIP.bAvieS8bvA8TPzHZb9EhIAHaEc?w=636&h=382&rs=1&pid=ImgDetMain",
    "descripcion": "Una mermelada exquisita hecha con cerezas frescas, ideal para acompañar pan, galletas o helados.",
    "ingredientes": ["Azúcar", "Pectina"],
    "pasos": [
        "Paso 1: Preparar las cerezas. Lava 500g de cerezas, quítales el hueso y córtalas por la mitad.",
        "Paso 2: Cocinar. Coloca las cerezas en una cacerola con 250g de azúcar y el jugo de 1 limón. Cocina a fuego medio hasta que se deshagan.",
        "Paso 3: Añadir pectina. Si prefieres una mermelada más espesa, agrega 1 cucharadita de pectina y cocina otros 10 minutos.",
        "Paso 4: Envasar. Vierte la mermelada en frascos esterilizados y ciérralos bien.",
        "Paso 5: Servir. Deja enfriar y disfruta de esta deliciosa mermelada de cereza sobre pan o galletas."
    ]
},
    "Tarta de cereza": {
    "imagen": "https://th.bing.com/th/id/OIP.k3eBQoqxewVB99aQTx_weAHaEK?rs=1&pid=ImgDetMain",
    "descripcion": "Una tarta fresca, con el toque ácido de las cerezas que la hace única y deliciosa.",
    "ingredientes": ["Harina", "Azúcar", "Mantequilla", "Huevos", "Leche"],
    "pasos": [
        "Paso 1: Preparar las cerezas. Lava 300g de cerezas y quítales el hueso.",
        "Paso 2: Preparar la masa. En un bol, mezcla 200g de harina, 150g de azúcar, 100g de mantequilla derretida y 2 huevos.",
        "Paso 3: Hornear la base. Vierte la mezcla en un molde engrasado y hornea a 180°C durante 25 minutos.",
        "Paso 4: Añadir las cerezas. Coloca las cerezas sobre la base horneada.",
        "Paso 5: Hornear nuevamente. Hornea durante 20 minutos más hasta que las cerezas estén suaves.",
        "Paso 6: Servir. Deja enfriar antes de servir."
    ]
},
    "Smoothie de cereza": {
    "imagen": "https://th.bing.com/th/id/R.f6ddc2152be3512e256aa6a7f1b483fe?rik=OSd707Hf13ijFg&pid=ImgRaw&r=0",
    "descripcion": "Un smoothie refrescante y lleno de antioxidantes, perfecto para un snack saludable o un desayuno rápido.",
    "ingredientes": ["Yogur", "Miel", "Hielo"],
    "pasos": [
        "Paso 1: Preparar las cerezas. Lava 150g de cerezas y retira los huesos.",
        "Paso 2: Licuar. Coloca las cerezas en una licuadora con 200g de yogur natural y 1 cucharadita de miel.",
        "Paso 3: Agregar hielo. Añade un puñado de hielo y licúa hasta obtener una mezcla suave.",
        "Paso 4: Servir. Sirve el smoothie de cereza en un vaso frío y disfruta."
    ]
},

    "Tarta de clementina": {
    "imagen": "https://th.bing.com/th/id/OIP.DksTlNoAU2mY_A2_dRXHvwAAAA?rs=1&pid=ImgDetMain",
    "descripcion": "Una tarta fresca con un toque cítrico delicioso, gracias a las clementinas. Perfecta para el postre o la merienda.",
    "ingredientes": [ "Harina", "Azúcar", "Mantequilla", "Huevos", "Leche"],
    "pasos": [
        "Paso 1: Preparar la masa. En un bol, mezcla 250g de harina, 150g de azúcar, 100g de mantequilla derretida y 2 huevos.",
        "Paso 2: Hornear la base. Vierte la mezcla en un molde y hornea a 180°C durante 25 minutos.",
        "Paso 3: Preparar la crema de clementina. Exprime el jugo de 3 clementinas y mezcla con 50g de azúcar y 200ml de leche. Cocina a fuego bajo hasta que espese.",
        "Paso 4: Rellenar la tarta. Vierte la crema sobre la base ya horneada.",
        "Paso 5: Dejar enfriar y servir. Deja que se enfríe antes de disfrutarla."
    ]
},
    "Ensalada de clementina": {
    "imagen": "https://th.bing.com/th/id/R.c80a9de7ff25647223d7f8f7fd8ec479?rik=6SRhQtmsTKJceg&pid=ImgRaw&r=0",
    "descripcion": "Una ensalada fresca y ligera, perfecta para acompañar cualquier plato. Las clementinas le dan un toque único.",
    "ingredientes": ["Lechuga", "Nueces", "Aceite de oliva", "Sal", "Pimienta"],
    "pasos": [
        "Paso 1: Pelar las clementinas. Pela 4 clementinas y córtalas en rodajas.",
        "Paso 2: Preparar la base. Coloca en un bol grande hojas de lechuga.",
        "Paso 3: Añadir las clementinas. Agrega las rodajas de clementina sobre la lechuga.",
        "Paso 4: Añadir nueces. Incorpora un puñado de nueces troceadas.",
        "Paso 5: Aliñar. Adereza con aceite de oliva, sal y pimienta al gusto. Sirve y disfruta."
    ]
},
    "Mermelada de clementina": {
    "imagen": "https://th.bing.com/th/id/OIP.sMP1jYtx2PsbiAU7eqTQGwHaHa?w=960&h=960&rs=1&pid=ImgDetMain",
    "descripcion": "Una mermelada aromática y deliciosa que captura toda la esencia de las clementinas, ideal para acompañar tostadas.",
    "ingredientes": ["Azúcar", "Pectina"],
    "pasos": [
        "Paso 1: Pelar las clementinas. Pela 500g de clementinas, quita las semillas y corta en trozos pequeños.",
        "Paso 2: Cocinar. Coloca las clementinas en una cacerola con 250g de azúcar y el jugo de 1 limón.",
        "Paso 3: Añadir pectina. Si deseas una mermelada espesa, agrega 1 cucharadita de pectina.",
        "Paso 4: Cocinar a fuego lento. Cocina por 30 minutos, removiendo constantemente.",
        "Paso 5: Envasar. Vierte la mermelada en frascos esterilizados y deja enfriar."
    ]
},

    "Tarta de coco": {
    "imagen": "https://th.bing.com/th/id/R.f8c76a9f3cd39e9b634c59c645b74e36?rik=QoUsjvBP5sfBDw&pid=ImgRaw&r=0",
    "descripcion": "Una tarta deliciosa con un sabor tropical, con la suavidad del coco y una base crujiente.",
    "ingredientes": ["Harina", "Azúcar", "Leche", "Huevos", "Mantequilla"],
    "pasos": [
        "Paso 1: Preparar la base. Mezcla 200g de harina con 100g de mantequilla derretida y 50g de azúcar.",
        "Paso 2: Hornear la base. Vierte la mezcla en un molde y hornea a 180°C durante 20 minutos.",
        "Paso 3: Preparar el relleno. En un bol, mezcla 200ml de leche, 100g de azúcar, 2 huevos y 150g de coco rallado.",
        "Paso 4: Rellenar la base. Vierte la mezcla sobre la base horneada.",
        "Paso 5: Hornear nuevamente. Hornea durante 30 minutos hasta que el relleno esté firme y dorado."
    ]
},
    "Bolitas de coco": {
    "imagen": "https://th.bing.com/th/id/R.15b7372e14f038084f3a889cd5071c4a?rik=k5lH%2fQeqGJebww&pid=ImgRaw&r=0",
    "descripcion": "Pequeñas bolitas deliciosas y dulces hechas de coco, perfectas como snack o postre.",
    "ingredientes": ["Leche condensada", "Azúcar"],
    "pasos": [
        "Paso 1: Mezclar ingredientes. En un bol, combina 200g de coco rallado con 150g de leche condensada y 50g de azúcar.",
        "Paso 2: Formar bolitas. Con las manos, forma pequeñas bolitas con la mezcla.",
        "Paso 3: Rebozar en coco. Pasa las bolitas por más coco rallado para cubrirlas.",
        "Paso 4: Refrigerar. Coloca las bolitas en la nevera por al menos 1 hora antes de servir."
    ]
},
    "Galletas de coco": {
    "imagen": "https://www.comedera.com/wp-content/uploads/sites/9/2023/02/Galletas-de-coco-caseras-shutterstock_1049204231.jpg",
    "descripcion": "Galletas crujientes y deliciosas con un toque de coco, perfectas para acompañar un café.",
    "ingredientes": ["Harina", "Azúcar", "Mantequilla", "Huevos"],
    "pasos": [
        "Paso 1: Preparar la masa. En un bol, mezcla 100g de mantequilla derretida, 150g de azúcar y 1 huevo.",
        "Paso 2: Añadir el coco. Agrega 150g de coco rallado y 200g de harina a la mezcla.",
        "Paso 3: Formar las galletas. Con las manos, forma pequeñas bolitas y aplástalas para darles forma de galleta.",
        "Paso 4: Hornear. Coloca las galletas en una bandeja y hornea a 180°C durante 15 minutos."
    ]
},
    "Ensalada de melón": {
    "imagen": "https://th.bing.com/th/id/OIP.4cwfcwk6iSPxJiT-O8l5WwHaE8?rs=1&pid=ImgDetMain",
    "descripcion": "Una ensalada fresca y ligera, ideal para los días calurosos. El melón le da un toque dulce y refrescante.",
    "ingredientes": ["Pepino", "Menta", "Jugo de limón"],
    "pasos": [
        "Paso 1: Cortar el melón. Pela y corta el melón en cubos pequeños.",
        "Paso 2: Preparar el pepino. Pela y corta 1 pepino en rodajas finas.",
        "Paso 3: Mezclar. Coloca el melón y el pepino en un bol grande.",
        "Paso 4: Añadir menta. Agrega unas hojas de menta picada.",
        "Paso 5: Aliñar. Exprime el jugo de 1 limón sobre la ensalada y mezcla bien."
    ]
},
    "Smoothie de melón": {
    "imagen": "https://www.codigosanluis.com/wp-content/uploads/2020/07/jugo-melon-800x533.jpg",
    "descripcion": "Un smoothie refrescante y saludable, perfecto para el verano. El melón le da una textura suave y natural.",
    "ingredientes": ["Yogur", "Miel", "Hielo"],
    "pasos": [
        "Paso 1: Cortar el melón. Corta el melón en trozos pequeños, eliminando las semillas.",
        "Paso 2: Licuar. Coloca el melón en una licuadora con 200g de yogur natural, 1 cucharadita de miel y un puñado de hielo.",
        "Paso 3: Licuar hasta obtener una mezcla suave y homogénea.",
        "Paso 4: Servir. Sirve el smoothie en un vaso frío y disfruta."
    ]
},
    "Tarta de melón": {
    "imagen": "https://th.bing.com/th/id/R.7e9cda80f61e4b53447367f2151b6db7?rik=3hdyFU%2fGhitTWw&pid=ImgRaw&r=0",
    "descripcion": "Una tarta fresca y suave con un sabor delicado a melón, perfecta para un día especial.",
    "ingredientes": ["Harina", "Azúcar", "Huevos", "Mantequilla"],
    "pasos": [
        "Paso 1: Preparar la masa. Mezcla 200g de harina con 100g de mantequilla derretida y 50g de azúcar.",
        "Paso 2: Hornear la base. Vierte la mezcla en un molde y hornea a 180°C durante 20 minutos.",
        "Paso 3: Preparar el relleno. En un bol, bate 2 huevos con 150g de azúcar y el puré de melón.",
        "Paso 4: Rellenar la tarta. Vierte la mezcla sobre la base horneada.",
        "Paso 5: Hornear nuevamente. Hornea durante 30 minutos."
    ]
},
    "Ensalada de melocotón": {
    "imagen": "https://th.bing.com/th/id/OIP.xdQxlChpmFUQcscshkuWQwAAAA?rs=1&pid=ImgDetMain",
    "descripcion": "Una ensalada fresca y afrutada con el toque dulce de los melocotones, ideal para acompañar carnes.",
    "ingredientes": ["Lechuga", "Queso de cabra", "Aceite de oliva"],
    "pasos": [
        "Paso 1: Cortar los melocotones. Pela y corta 3 melocotones en rodajas finas.",
        "Paso 2: Preparar la base. Coloca hojas de lechuga en un bol.",
        "Paso 3: Añadir melocotón. Coloca las rodajas de melocotón sobre la lechuga.",
        "Paso 4: Añadir queso de cabra. Desmenuza queso de cabra y agrégalo a la ensalada.",
        "Paso 5: Aliñar. Adereza con aceite de oliva, sal y pimienta al gusto."
    ]
},
    "Tarta de melocotón": {
    "imagen": "https://th.bing.com/th/id/OIP.05mALW9xUYlNi7UmJpHmdQHaFj?rs=1&pid=ImgDetMain",
    "descripcion": "Una deliciosa tarta con el sabor suave y dulce del melocotón, perfecta para cualquier ocasión.",
    "ingredientes": ["Harina", "Azúcar", "Huevos", "Mantequilla"],
    "pasos": [
        "Paso 1: Preparar la masa. Mezcla 200g de harina con 100g de mantequilla derretida y 50g de azúcar.",
        "Paso 2: Hornear la base. Vierte la mezcla en un molde y hornea a 180°C durante 20 minutos.",
        "Paso 3: Preparar el relleno. Pela y corta 3 melocotones en rodajas finas.",
        "Paso 4: Colocar las rodajas. Coloca las rodajas de melocotón sobre la base.",
        "Paso 5: Hornear nuevamente. Hornea durante 30 minutos hasta que esté dorado."
    ]
},
    "Mermelada de melocotón": {
    "imagen": "https://www.conasi.eu/blog/wp-content/uploads/2012/09/mermelada-de-melocot%C3%B3n-1.2.jpg",
    "descripcion": "Una mermelada suave, dulce y afrutada, perfecta para acompañar pan o galletas.",
    "ingredientes": ["Azúcar", "Limón" "Pectina"],
    "pasos": [
        "Paso 1: Preparar los melocotones. Pela y corta 500g de melocotones.",
        "Paso 2: Cocinar. Coloca los melocotones en una cacerola con 250g de azúcar y el jugo de 1 limón.",
        "Paso 3: Añadir pectina. Agrega 1 cucharadita de pectina si deseas una textura más espesa.",
        "Paso 4: Cocinar a fuego lento. Cocina durante 30 minutos, removiendo constantemente.",
        "Paso 5: Envasar. Coloca la mermelada en frascos y deja enfriar."
    ]
},

    "Jugo de carambola": {
    "imagen": "https://2.bp.blogspot.com/-0pcoHJRyFn0/WQsuMxwlZOI/AAAAAAAAAX4/4x2BJKPm95Y5M9hJyQojSDtS9zur_-APQCLcB/s640/0457.png",
    "descripcion": "Un jugo refrescante con un sabor único y tropical gracias a la carambola.",
    "ingredientes": ["Azúcar", "Agua", "Hielo"],
    "pasos": [
        "Paso 1: Preparar las carambolas. Lava y corta 2 carambolas en rodajas finas.",
        "Paso 2: Licuar. Coloca las rodajas de carambola en la licuadora con 500ml de agua y 2 cucharadas de azúcar.",
        "Paso 3: Licuar. Licúa hasta obtener un líquido suave y homogéneo.",
        "Paso 4: Servir. Vierte el jugo en un vaso con hielo y disfruta."
    ]
},

    "Ensalada de carambola": {
    "imagen": "https://th.bing.com/th/id/OIP.HmsQ_3NWHZuWyNzHMDUjAAHaFh?rs=1&pid=ImgDetMain",
    "descripcion": "Una ensalada fresca y exótica con el toque dulce y ácido de la carambola.",
    "ingredientes": ["Pepino", "Menta", "Aceite de oliva", "Limón"],
    "pasos": [
        "Paso 1: Cortar la carambola. Lava y corta 2 carambolas en rodajas finas.",
        "Paso 2: Cortar el pepino. Pela y corta 1 pepino en rodajas finas.",
        "Paso 3: Preparar la ensalada. Coloca las rodajas de carambola y pepino en un bol.",
        "Paso 4: Añadir menta. Agrega unas hojas de menta fresca.",
        "Paso 5: Aliñar. Exprime el jugo de 1 limón y adereza con aceite de oliva al gusto."
    ]
},

    "Mermelada de carambola": {
    "imagen": "https://th.bing.com/th/id/OIP.HnL8R8wK7FTOt2oVndb27gHaEK?rs=1&pid=ImgDetMain",
    "descripcion": "Una mermelada deliciosa y tropical, perfecta para acompañar pan tostado o galletas.",
    "ingredientes": ["Azúcar", "Limón", "Pectina"],
    "pasos": [
        "Paso 1: Preparar las carambolas. Lava y corta 500g de carambola en trozos pequeños.",
        "Paso 2: Cocinar. Coloca las carambolas en una cacerola con 250g de azúcar y el jugo de 1 limón.",
        "Paso 3: Añadir pectina. Agrega 1 cucharadita de pectina para espesar la mezcla.",
        "Paso 4: Cocinar a fuego lento. Cocina por 30 minutos hasta que espese.",
        "Paso 5: Envasar. Coloca la mermelada en frascos esterilizados y deja enfriar."
    ]
},
    "Smoothie de cherimoya": {
    "imagen": "https://th.bing.com/th/id/OIP.-D9tnM7ARgtAI34CndnJxAHaFR?w=1024&h=729&rs=1&pid=ImgDetMain",
    "descripcion": "Un smoothie tropical y cremoso, ideal para un desayuno o merienda.",
    "ingredientes": ["Leche", "Miel", "Hielo"],
    "pasos": [
        "Paso 1: Preparar la cherimoya. Pela y saca la pulpa de 1 cherimoya.",
        "Paso 2: Licuar. Coloca la pulpa en la licuadora con 200ml de leche y 1 cucharadita de miel.",
        "Paso 3: Añadir hielo. Agrega un puñado de hielo y licúa hasta que esté suave.",
        "Paso 4: Servir. Sirve en un vaso frío y disfruta."
    ]
},
    "Ensalada de cherimoya": {
    "imagen": "https://th.bing.com/th/id/OIP.3duQltdsmJGgjk09-tFN_AHaE7?rs=1&pid=ImgDetMain",
    "descripcion": "Una ensalada fresca y exótica, con la dulzura natural de la cherimoya.",
    "ingredientes": ["Menta", "Lima"],
    "pasos": [
        "Paso 1: Preparar la cherimoya. Pela y corta 1 cherimoya en cubos.",
        "Paso 2: Cortar la piña. Pela y corta 1/2 piña en trozos pequeños.",
        "Paso 3: Mezclar. Coloca la cherimoya y la piña en un bol grande.",
        "Paso 4: Añadir menta. Incorpora unas hojas de menta fresca.",
        "Paso 5: Aliñar. Exprime el jugo de 1 lima y mezcla bien."
    ]
},
    "Mermelada de cherimoya": {
    "imagen": "https://th.bing.com/th/id/OIP._8Ji4gch-6rOEygE47aA4QHaE7?rs=1&pid=ImgDetMain",
    "descripcion": "Una mermelada suave y dulce, ideal para acompañar tostadas.",
    "ingredientes": ["Azúcar", "Limón", "Pectina"],
    "pasos": [
        "Paso 1: Preparar la cherimoya. Pela y corta 500g de cherimoya.",
        "Paso 2: Cocinar. Coloca la pulpa en una cacerola con 250g de azúcar y el jugo de 1 limón.",
        "Paso 3: Añadir pectina. Agrega 1 cucharadita de pectina para espesar la mermelada.",
        "Paso 4: Cocinar a fuego lento. Cocina durante 30 minutos, removiendo ocasionalmente.",
        "Paso 5: Envasar. Coloca la mermelada en frascos y deja enfriar."
    ]
},

    "Ensalada de uva y queso": {
    "imagen": "https://th.bing.com/th/id/OIP.IKt4aaMc4WheGfRYbTRDsAHaE8?rs=1&pid=ImgDetMain",
    "descripcion": "Una ensalada fresca y deliciosa que combina el dulzor de las uvas con la cremosidad del queso. Perfecta para acompañar platos principales o como aperitivo.",
    "ingredientes": ["Queso fresco", "Lechuga", "Vinagre balsámico", "Aceite de oliva"],
    "pasos": [
        "Paso 1: Preparar los ingredientes. Lava y corta por la mitad 200g de uvas. Corta 150g de queso fresco en cubos pequeños.",
        "Paso 2: Mezclar la ensalada. En un bol grande, coloca las uvas, el queso fresco y unas hojas de lechuga fresca y lavada.",
        "Paso 3: Aliñar. En un pequeño tazón, mezcla 2 cucharadas de vinagre balsámico con 3 cucharadas de aceite de oliva. Vierte sobre la ensalada y mezcla suavemente.",
        "Paso 4: Ajustar el sabor. Sazona con sal y pimienta al gusto. Si deseas un toque más dulce, añade una cucharadita de miel.",
        "Paso 5: Servir. Sirve esta ensalada fresca como acompañante de un plato principal o disfrútala sola como un snack saludable."
    ]
},
    "Ensalada de uva y queso feta": {
        "imagen": "https://th.bing.com/th/id/OIP.8aKROVV0RGaUvwSO3wJASwHaE8?rs=1&pid=ImgDetMain",
        "descripcion": "Una ensalada fresca con uvas y el toque salado del queso feta.",
        "ingredientes": ["Queso feta", "Nueces", "Aceite de oliva", "Miel"],
        "pasos": [
            "Paso 1: Preparar las uvas. Lava y corta las uvas por la mitad.",
            "Paso 2: Cortar el queso. Corta 100g de queso feta en cubos pequeños.",
            "Paso 3: Mezclar. Coloca las uvas y el queso en un bol, y agrega un puñado de nueces.",
            "Paso 4: Aliñar. Rocía con aceite de oliva y miel al gusto.",
            "Paso 5: Servir. Mezcla bien y disfruta de una ensalada refrescante."
        ]
    },
    "Mermelada de uva": {
        "imagen": "https://www.deliciosamermelada.com/wp-content/uploads/2016/07/receta-de-mermelada-de-uva-casera-730x430.jpg",
        "descripcion": "Una mermelada dulce y suave para untar sobre pan o acompañar postres.",
        "ingredientes": ["Azúcar", "Jugo de limón", "Pectina"],
        "pasos": [
            "Paso 1: Preparar las uvas. Lava y corta las uvas a la mitad.",
            "Paso 2: Cocinar. Coloca las uvas en una cacerola con 300g de azúcar y el jugo de 1 limón.",
            "Paso 3: Añadir pectina. Agrega 1 cucharadita de pectina.",
            "Paso 4: Cocinar a fuego lento. Cocina durante 30-40 minutos, removiendo de vez en cuando.",
            "Paso 5: Envasar. Vierte la mermelada caliente en frascos esterilizados y deja enfriar."
        ]
    },
    "Tarta de uva": {
        "imagen": "URL_DE_LA_IMAGEN_AQUI",
        "descripcion": "Una deliciosa tarta con una base crujiente y el dulce sabor de las uvas.",
        "ingredientes": ["Harina", "Azúcar", "Mantequilla", "Huevos"],
        "pasos": [
            "Paso 1: Preparar la base. Mezcla 250g de harina con 125g de mantequilla derretida y 100g de azúcar.",
            "Paso 2: Amasar. Amasa hasta obtener una masa homogénea y colócala en un molde para tartas.",
            "Paso 3: Preparar el relleno. Bate 2 huevos con 100g de azúcar y 200ml de nata.",
            "Paso 4: Colocar las uvas. Coloca las uvas sobre la base de la tarta y vierte el relleno encima.",
            "Paso 5: Hornear. Hornea a 180°C durante 40 minutos."
        ]
    },

    "Mermelada de guayaba": {
        "imagen": "https://recetacubana.com/wp-content/uploads/2018/11/mermelada-de-guayaba.jpg",
        "descripcion": "Una mermelada dulce y tropical perfecta para untar en pan o acompañar postres.",
        "ingredientes": ["Azúcar", "Jugo de limón", "Pectina"],
        "pasos": [
            "Paso 1: Preparar las guayabas. Pela y corta las guayabas en trozos pequeños.",
            "Paso 2: Cocinar. Coloca las guayabas en una cacerola con 300g de azúcar y el jugo de 1 limón.",
            "Paso 3: Añadir pectina. Agrega 1 cucharadita de pectina.",
            "Paso 4: Cocinar a fuego lento. Cocina durante 30-40 minutos, removiendo de vez en cuando.",
            "Paso 5: Envasar. Vierte la mermelada caliente en frascos esterilizados y deja enfriar."
        ]
    },
    "Tarta de guayaba": {
        "imagen": "https://th.bing.com/th/id/OIP.5SEM8UI2Pa4HMTC4BU8vvwHaE8?rs=1&pid=ImgDetMain",
        "descripcion": "Una tarta suave y esponjosa con el sabor dulce de la guayaba.",
        "ingredientes": ["Harina", "Azúcar", "Mantequilla", "Huevos"],
        "pasos": [
            "Paso 1: Preparar la base. Mezcla 250g de harina con 125g de mantequilla derretida y 100g de azúcar.",
            "Paso 2: Amasar. Amasa hasta obtener una masa homogénea y colócala en un molde para tartas.",
            "Paso 3: Preparar el relleno. Bate 2 huevos con 100g de azúcar y 200ml de nata.",
            "Paso 4: Colocar las guayabas. Coloca las guayabas sobre la base de la tarta y vierte el relleno encima.",
            "Paso 5: Hornear. Hornea a 180°C durante 40 minutos."
        ]
    },
    "Batido de guayaba": {
        "imagen": "https://th.bing.com/th/id/OIP.wSAqBxv5rOgF680gOVZvbQHaEk?rs=1&pid=ImgDetMain",
        "descripcion": "Un batido tropical y refrescante con el sabor dulce y ácido de la guayaba.",
        "ingredientes": ["Leche", "Azúcar", "Hielo"],
        "pasos": [
            "Paso 1: Preparar las guayabas. Pela y corta 2 guayabas en trozos.",
            "Paso 2: Licuar. Coloca las guayabas en una licuadora con 200ml de leche, 1 cucharadita de azúcar y un puñado de hielo.",
            "Paso 3: Licuar. Licúa hasta que todo esté bien mezclado.",
            "Paso 4: Servir. Vierte en un vaso y disfruta de este delicioso batido."
        ]
},

      "Ensalada de kiwi": {
        "imagen": "URL_DE_LA_IMAGEN_AQUI",
        "descripcion": "Una ensalada fresca con el toque ácido del kiwi.",
        "ingredientes": ["Menta", "Aceite de oliva"],
        "pasos": [
            "Paso 1: Pelar y cortar 3 kiwis en rodajas finas.",
            "Paso 2: Colocar los kiwis en un bol y añadir unas hojas de menta fresca.",
            "Paso 3: Rocíe con un poco de aceite de oliva y mezcle suavemente.",
            "Paso 4: Servir inmediatamente y disfrutar de esta ensalada refrescante."
        ]
    },
    "Tarta de kiwi": {
        "imagen": "https://th.bing.com/th/id/R.55bacc192e38f501078576493362795b?rik=WwIop5NeySJ6Hw&pid=ImgRaw&r=0",
        "descripcion": "Una tarta fresca y colorida con el toque ácido del kiwi.",
        "ingredientes": ["Masa de tarta", "Crema pastelera", "Azúcar"],
        "pasos": [
            "Paso 1: Hornea la masa de tarta siguiendo las instrucciones del paquete.",
            "Paso 2: Prepara la crema pastelera: cocina 500ml de leche con 4 yemas de huevo, 100g de azúcar y 1 cucharadita de maicena.",
            "Paso 3: Una vez que la base de la tarta esté enfriada, cubre con la crema pastelera.",
            "Paso 4: Decora la tarta con rodajas de kiwi sobre la crema pastelera.",
            "Paso 5: Refrigera la tarta por unas horas y sirve fría."
        ]
    },
    "Sorbet de kiwi": {
        "imagen": "https://th.bing.com/th/id/OIP.OC0hbg02D93SVizKNDTPvgHaFj?rs=1&pid=ImgDetMain",
        "descripcion": "Un sorbete refrescante y dulce con el sabor único del kiwi.",
        "ingredientes": ["Azúcar", "Agua", "Hielo"],
        "pasos": [
            "Paso 1: Pela y corta 5 kiwis en trozos pequeños.",
            "Paso 2: Licúa los kiwis con 200ml de agua y 100g de azúcar.",
            "Paso 3: Vierte la mezcla en una bandeja y congela por 4 horas, removiendo cada 30 minutos.",
            "Paso 4: Sirve el sorbete en copas y disfruta de un postre refrescante."
        ]
    },

    "Mermelada de mandarina": {
        "imagen": "https://th.bing.com/th/id/R.70786b94c3f1494b456aa7802f71c51d?rik=C89kWdAI8zx9Aw&pid=ImgRaw&r=0",
        "descripcion": "Una mermelada deliciosa y ácida, ideal para untar en pan.",
        "ingredientes": ["Azúcar", "Pectina", "Jugo de limón"],
        "pasos": [
            "Paso 1: Pela y corta 5 mandarinas en trozos.",
            "Paso 2: Coloca las mandarinas en una cacerola con 300g de azúcar y el jugo de 1 limón.",
            "Paso 3: Añadir 1 cucharadita de pectina.",
            "Paso 4: Cocina a fuego lento durante 30-40 minutos, removiendo de vez en cuando.",
            "Paso 5: Vierte la mermelada caliente en frascos esterilizados y deja enfriar."
        ]
    },
    "Ensalada de mandarina": {
        "imagen": "https://th.bing.com/th/id/OIP.G3w9WD4ERU8EB7OlhZylBwHaE8?rs=1&pid=ImgDetMain",
        "descripcion": "Una ensalada fresca con gajos de mandarina.",
        "ingredientes": ["Lechuga", "Aceite de oliva", "Sal"],
        "pasos": [
            "Paso 1: Pela y divide las mandarinas en gajos.",
            "Paso 2: Lava las hojas de lechuga y colócalas en un bol.",
            "Paso 3: Añade los gajos de mandarina a la lechuga.",
            "Paso 4: Aliña con aceite de oliva y sal al gusto.",
            "Paso 5: Mezcla bien y disfruta."
        ]
    },
    "Jugo de mandarina": {
        "imagen": "https://corp.ametllerorigen.com/wp-content/uploads/2021/01/ZUMO-AUMENTA-DEFENSAS-ZUMO-DE-MANDARINA-CON-CANELA-Y-POLEN-Ametller-Origen.jpg",
        "descripcion": "Un jugo natural y refrescante para disfrutar durante todo el día.",
        "ingredientes": ["Azúcar (opcional)"],
        "pasos": [
            "Paso 1: Exprime el jugo de 6 mandarinas.",
            "Paso 2: Si lo prefieres más dulce, agrega 1 cucharadita de azúcar.",
            "Paso 3: Mezcla bien y sirve frío o con hielo."
        ]
}

}
# Mapeo de ingredientes a recetas
ingrediente_a_receta = {
    "Manzanas": ["Tarta de manzana", "Compota de manzana", "Tarta Tatín de manzana"],
    "Albaricoque": ["Mermelada de albaricoque", "Tarta de albaricoque", "Ensalada de albaricoque y queso de cabra"],
    "Aguacate": ["Guacamole","Tostadas de aguacate con huevo poché", "Ensalada de aguacate y tomate con vinagreta de mostaza"],
    "Plátano": ["Pan de plátano", "Smoothie de plátano", "Tarta de plátano"],
    "Mora": ["Mermelada de mora", "Tarta de mora", "Smoothie de mora"],
    "Arándano": ["Mermelada de arándano", "Tarta de arándano", "Smoothie de arándano"],
    "Cereza": ["Mermelada de cereza", "Tarta de cereza", "Smoothie de cereza"],
    "Clementina": ["Tarta de clementina", "Ensalada de clementina", "Mermelada de clementina"],
    "Coco": ["Tarta de coco", "Bolitas de coco", "Galletas de coco"],
    "Melón": ["Ensalada de melón", "Smoothie de melón", "Tarta de melón"],
    "Melocotón": ["Ensalada de melocotón", "Tarta de melocotón", "Mermelada de melocotón"],
    "Carambola": ["Jugo de carambola", "Ensalada de carambola",],
    "Cherimoya": ["Smoothie de cherimoya", "Ensalada de cherimoya", "Mermelada de cherimoya"],
    "Uva":  ["Ensalada de uva y queso", "Ensalada de uva y queso feta", "Mermelada de uva"],
    "Guayaba": ["Mermelada de guayaba", "Tarta de guayaba", "Batido de guayaba"],
    "Kiwi": ["Ensalada de kiwi", "Tarta de kiwi",   "Sorbet de kiwi"],
    "Mandarina": [ "Mermelada de mandarina",  "Ensalada de mandarina",  "Jugo de mandarina"]
}

# Estado de la receta actual para cada ingrediente
if "receta_actual" not in st.session_state:
    st.session_state.receta_actual = {ing: 0 for ing in ingrediente_a_receta}

# Estado de lectura en voz alta
if "leyendo_receta" not in st.session_state:
    st.session_state.leyendo_receta = {ing: False for ing in ingrediente_a_receta}

# Función para sugerir recetas
def sugerir_recetas(ingredientes_disponibles):
    recetas_sugeridas = {}
    for nombre_receta, datos in todas_las_recetas.items():
        ingredientes_faltantes = [ing for ing in datos["ingredientes"] if ing not in ingredientes_disponibles]
        recetas_sugeridas[nombre_receta] = {
            "descripcion": datos["descripcion"],
            "pasos": datos["pasos"],
            "imagen": datos["imagen"],
            "faltantes": ingredientes_faltantes
        }
    return recetas_sugeridas

# Obtener recetas según ingredientes seleccionados
ingredientes_seleccionados = st.multiselect("Selecciona los ingredientes disponibles:", list(ingrediente_a_receta.keys()))
recetas_sugeridas = sugerir_recetas(ingredientes_seleccionados)

# Función para leer la receta en voz alta sin guardar archivos en disco
def leer_receta(ingrediente, receta):
    datos_receta = todas_las_recetas[receta]
    texto = f"Receta: {receta}. Descripción: {datos_receta['descripcion']}. Pasos: "
    for i, paso in enumerate(datos_receta["pasos"], start=1):
        texto += f"Paso {i}: {paso}. "

    # Generar audio con gTTS en memoria
    tts = gTTS(texto, lang='es')
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)

    # Convertir audio a base64 para Streamlit
    audio_base64 = base64.b64encode(audio_bytes.read()).decode()
    st.session_state.leyendo_receta[ingrediente] = audio_base64

# Mostrar recetas disponibles
if ingredientes_seleccionados:
    for ingrediente in ingredientes_seleccionados:
        receta_idx = st.session_state.receta_actual[ingrediente]
        receta_nombre = ingrediente_a_receta[ingrediente][receta_idx]
        receta = recetas_sugeridas[receta_nombre]

        # Mostrar detalles de la receta
        st.subheader(f"{receta_nombre}")
        st.write(f"URL de la imagen: {receta.get('imagen', 'No disponible')}")

        # Verificar si la clave 'imagen' existe y no es None o vacía
        if isinstance(receta.get("imagen"), str) and receta["imagen"].strip():
         st.image(receta["imagen"], caption=receta_nombre, use_container_width=True)
        else:
            st.warning(f"⚠️ No se encontró una imagen para {receta_nombre}")
        st.write(f"**Descripción:** {receta['descripcion']}")
        st.write("**Pasos:**")
        for i, paso in enumerate(receta["pasos"], start=1):
            st.write(f"**Paso {i}:** {paso}")


        # Mostrar ingredientes faltantes
        if receta["faltantes"]:
            st.warning(f"⚠️ Te faltan estos ingredientes: {', '.join(receta['faltantes'])}")
        else:
            st.success("✅ Tienes todos los ingredientes para esta receta")

        # Botón para cambiar la receta del ingrediente
        if st.button(f"🔄 Cambiar receta de {ingrediente}"):
            st.session_state.receta_actual[ingrediente] = (receta_idx + 1) % len(ingrediente_a_receta[ingrediente])
            st.rerun()

        # Botón para leer la receta
        if st.button(f"📖 (Por favor espera unos minutos) Leer la receta de {receta_nombre}"):
            leer_receta(ingrediente, receta_nombre)

        # Si hay audio generado, mostrar reproductor
        if st.session_state.leyendo_receta[ingrediente]:
            st.audio(io.BytesIO(base64.b64decode(st.session_state.leyendo_receta[ingrediente])), format="audio/mp3")

st.write("---")

# Botón para cambiar todas las recetas al siguiente tipo
if st.button("🔄 Otra receta para todos"):
    for ingrediente in ingredientes_seleccionados:
        st.session_state.receta_actual[ingrediente] = (st.session_state.receta_actual[ingrediente] + 1) % len(ingrediente_a_receta[ingrediente])
    st.rerun()
