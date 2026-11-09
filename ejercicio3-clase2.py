import asyncio
from typing import List
from pydantic import BaseModel, Field, ValidationError

# (dotenv): función que lee el archivo .env y carga sus líneas como variables de entorno
from dotenv import load_dotenv

# Errores de transporte del SDK de OpenAI (los que sí conviene reintentar / derivar a otro modelo)
from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  # Conector de Gemini
from langchain_core.prompts import ChatPromptTemplate

# (dotenv): carga OPENAI_API_KEY y GOOGLE_API_KEY desde el archivo .env.
# Tiene que ir ANTES de crear los modelos, porque ellos buscan las claves
# en el entorno al momento de construirse.
load_dotenv()


# --- ESQUEMA ---------------------------------------------------------------
class EntityExtraction(BaseModel):
    topic: str = Field(description="Tema principal del texto")
    entities: List[str] = Field(
        description="Personas, organizaciones, productos o tecnologías mencionadas"
    )
    sentiment_score: float = Field(
        ge=0, le=1,  # regla de negocio: debe estar entre 0 y 1
        description="Sentimiento del texto: 0 es muy negativo, 1 es muy positivo",
    )


# Tupla con los errores "de transporte". Los usamos para decidir
# cuándo reintentar y cuándo pasar a Gemini. Un ValidationError NO está acá,
# así que nunca activa ni el reintento ni el fallback.
TRANSPORT_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


# --- MODELO PRINCIPAL (OpenAI) ---------------------------------------------
# max_retries=0 desactiva los reintentos internos del SDK de OpenAI
# (por defecto hace 2), para que el único que reintente sea with_retry y
# no se multipliquen los intentos.
llm = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=0)

# Salida estructurada: devuelve un objeto EntityExtraction validado.
structured_llm = llm.with_structured_output(EntityExtraction)

# Solo reintenta errores de transporte (hasta 3 intentos, con backoff).
# Un ValidationError es un error lógico: reintentarlo solo gasta plata y latencia.
resilient_llm = structured_llm.with_retry(
    retry_if_exception_type=TRANSPORT_ERRORS,
    stop_after_attempt=3,
)


# --- MODELO DE RESPALDO (Gemini) -------------------------------------------
# Ajustá el nombre del modelo al que tengas disponible en tu cuenta de Google.
gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Gemini también necesita su propia salida estructurada con el mismo esquema,
# para que el resultado tenga la misma forma sin importar quién responda.
gemini_structured = gemini.with_structured_output(EntityExtraction)

# Si gpt-4o sigue fallando con un error de transporte después de los
# 3 intentos, se llama a Gemini. exceptions_to_handle limita el fallback a esos
# errores: un ValidationError no pasa a Gemini, se propaga al except de abajo.
llm_with_fallback = resilient_llm.with_fallbacks(
    [gemini_structured],
    exceptions_to_handle=TRANSPORT_ERRORS,
)


# --- PROMPT Y CADENA -------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extraé la información estructurada del texto del usuario."),
    ("human", "{input}"),
])

# La cadena usa el modelo con reintentos + fallback.
chain = prompt | llm_with_fallback

sample_text = (
    "LangGraph es una librería de LangChain para construir agentes con estado. "
    "Me parece muy útil, aunque la documentación todavía tiene huecos."
)


# --- EJECUCIÓN -------------------------------------------------------------
async def main():
    try:
        result = await chain.ainvoke({"input": sample_text})
        print(result.model_dump_json(indent=2))

    # El modelo (OpenAI o Gemini) respondió pero no respetó el esquema.
    except ValidationError as e:
        print("Error de validación (el modelo no respetó el esquema):", type(e).__name__)

    # Fallaron OpenAI (con sus reintentos) Y Gemini, o hubo otro error (API key inválida, etc.).
    except Exception as e:
        print("Fallaron todos los modelos u otro error:", type(e).__name__, "-", e)


asyncio.run(main())