"""
Ejercicio 1: reseñas (paralelo -> síntesis)

Por cada reseña:
  1. En paralelo: se analiza el sentimiento y se extrae el tema principal.
  2. Con ambos resultados se escribe una línea de ticket de soporte.

Son 3 llamadas al modelo por reseña. Se procesan varias reseñas a la vez con batch.

Requisitos:
    pip install langchain-openai python-dotenv
    OPENAI_API_KEY en el archivo .env
"""

# (dotenv): lee el archivo .env y carga sus líneas como variables de entorno
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Tiene que ir ANTES de crear el modelo: busca OPENAI_API_KEY en el entorno al construirse.
load_dotenv()

# --- COMPONENTES BASE ------------------------------------------------------
model = ChatOpenAI(model="gpt-4o", temperature=0)
parser = StrOutputParser()

# --- DOS CADENAS QUE CORREN EN PARALELO ------------------------------------
sentiment_chain = (
    ChatPromptTemplate.from_template(
        "Analizá el sentimiento de la siguiente reseña. "
        "Respondé con una sola palabra: positivo, negativo o neutro.\n\nReseña: {review}"
    )
    | model
    | parser
)

topic_chain = (
    ChatPromptTemplate.from_template(
        "Indicá cuál es el tema principal de la siguiente reseña "
        "en pocas palabras (máximo 5).\n\nReseña: {review}"
    )
    | model
    | parser
)

# RunnableParallel ejecuta ambas cadenas al mismo tiempo con la misma entrada
# y devuelve un dict: {"sentiment": ..., "topic": ...}
analysis = RunnableParallel(sentiment=sentiment_chain, topic=topic_chain)

# --- SÍNTESIS --------------------------------------------------------------
# Las claves del dict que sale de `analysis` (sentiment, topic) deben coincidir
# con las variables de este prompt.
ticket_prompt = ChatPromptTemplate.from_template(
    "Sentimiento: {sentiment}\n"
    "Tema: {topic}\n"
    "Escribí una línea de ticket de soporte para el equipo, clara y accionable."
)

full_chain = analysis | ticket_prompt | model | parser

# --- EJECUCIÓN -------------------------------------------------------------
reviews = [
    "El envío llegó tres días tarde y la caja estaba rota, muy decepcionante.",
    "Excelente atención al cliente, resolvieron mi consulta en minutos.",
    "La app se cierra sola cada vez que intento pagar con tarjeta.",
    "El producto es correcto, hace lo que dice, nada más.",
]

if __name__ == "__main__":
    # batch recibe una lista de inputs y devuelve una lista de resultados EN EL MISMO ORDEN.
    # max_concurrency limita cuántas cadenas corren a la vez (evita rate limits).
    tickets = full_chain.batch(
        [{"review": r} for r in reviews],
        config={"max_concurrency": 5},
    )

    for review, ticket in zip(reviews, tickets):
        print(f"Reseña: {review}")
        print(f"Ticket: {ticket}")
        print("-" * 60)
