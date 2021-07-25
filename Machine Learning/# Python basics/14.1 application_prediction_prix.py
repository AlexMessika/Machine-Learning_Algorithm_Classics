#1. Télecharger ce fichier et le fichier regression.joblib plus bas
#2. installer streamlit : pip install streamlit
#3. Lancer l

import streamlit as st
import joblib
import numpy as np

model = joblib.load('regression.joblib')


st.title('Prediction de prix de maison')

st.subheader('Entrez les caractéristiques de votre maison')

size = st.number_input('Taille de la maison')
nb_room = st.number_input('Nombre de chambres')
garden = st.number_input('Y a t-il un jardin ?')


if size == 0 or nb_room == 0:

    st.write("Compléter la taille et le nombre de chambre")

else:

    X = np.array([[size, nb_room, garden]])

    y = model.predict(X)

    st.write('La maison a pour prix {}'.format(y[0]))
