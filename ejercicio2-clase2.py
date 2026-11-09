"""
Ejercicio 2: cadena secuencial con lambda

Paso 1: genera una descripción de 2 oraciones para un producto.
Paso 2: resume esa descripción en un tweet con 1 emoji.

Una lambda convierte el str que sale del paso 1 en el dict que espera el paso 2.
2 productos x 2 pasos = 4 llamadas al modelo.

Requisitos:
    pip install langchain-openai python-dotenv
    OPENAI_API_KEY en el archivo .env
"""

# (dotenv): lee el archivo .env y carga sus líneas como variables de entorno
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Tiene que ir ANTES de crear el modelo: busca OPENAI_API_KEY en el entorno al construirse.
load_dotenv()

# --- COMPONENTES BASE ------------------------------------------------------
model = ChatOpenAI(model="gpt-4o", temperature=0)
parser = StrOutputParser()

# --- PASO 1: descripción ---------------------------------------------------
# Entrada: {"producto": ...}  ->  Salida: str
descripcion_chain = (
    ChatPromptTemplate.from_template("Descripción de 2 oraciones para: {producto}")
    | model
    | parser
)

# --- PASO 2: tweet ---------------------------------------------------------
# Entrada: {"descripcion": ...}  ->  Salida: str
tweet_chain = (
    ChatPromptTemplate.from_template("Resumí en un tweet con 1 emoji: {descripcion}")
    | model
    | parser
)

# --- CADENA COMPLETA -------------------------------------------------------
# La lambda es un Runnable: adapta el str del paso 1 al dict que necesita el paso 2.
full_chain = descripcion_chain | (lambda desc: {"descripcion": desc}) | tweet_chain

# --- EJECUCIÓN -------------------------------------------------------------
productos = [
    {"producto": "termo de acero 1L"},
    {"producto": "mochila antirrobo"},
]

if __name__ == "__main__":
    # batch mantiene el orden, por eso zip puede emparejar cada producto con su tweet.
    tweets = full_chain.batch(productos)

    for p, t in zip(productos, tweets):
        print(f"{p['producto']}: {t}")
