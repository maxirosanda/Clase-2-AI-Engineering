import asyncio
from typing import List
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv  # lee el .env y carga las claves

# Errores de transporte del SDK de OpenAI (los que conviene
# reintentar o derivar a otro modelo)
from openai import (APIConnectionError, APITimeoutError,
                    RateLimitError, InternalServerError)
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini
from langchain_core.prompts import ChatPromptTemplate

# ANTES de crear los modelos: ellos buscan OPENAI_API_KEY y
# GOOGLE_API_KEY en el entorno al construirse
load_dotenv()

# --- ESQUEMA ---
class EntityExtraction(BaseModel):
    topic: str = Field(description="Tema principal del texto")
    entities: List[str] = Field(
        description="Personas, organizaciones, productos o tecnologías")
    sentiment_score: float = Field(
        ge=0, le=1,  # regla de negocio: entre 0 y 1
        description="0 es muy negativo, 1 es muy positivo")

# Errores "de transporte": deciden cuándo reintentar y cuándo
# pasar a Gemini. Un ValidationError NO está acá, así que nunca
# activa ni el reintento ni el fallback.
TRANSPORT_ERRORS = (APIConnectionError, APITimeoutError,
                    RateLimitError, InternalServerError)

# --- MODELO PRINCIPAL (OpenAI) ---
# max_retries=0: desactiva los reintentos internos del SDK
# (por defecto hace 2) para que el único que reintente sea
# with_retry y no se multipliquen los intentos
llm = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=0)

# Salida estructurada: devuelve un objeto EntityExtraction validado
structured_llm = llm.with_structured_output(EntityExtraction)

# Reintenta solo errores de transporte (hasta 3 intentos, con
# backoff). Un ValidationError es un error lógico: reintentarlo
# solo gasta plata y latencia.
resilient_llm = structured_llm.with_retry(
    retry_if_exception_type=TRANSPORT_ERRORS,
    stop_after_attempt=3)

# --- MODELO DE RESPALDO (Gemini) ---
# Ajustá el modelo al que tengas disponible en tu cuenta
gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Mismo esquema: el resultado tiene la misma forma
# sin importar qué modelo responda
gemini_structured = gemini.with_structured_output(EntityExtraction)

# Si gpt-4o sigue fallando con un error de transporte tras los
# 3 intentos, se llama a Gemini. exceptions_to_handle limita el
# fallback a esos errores: un ValidationError se propaga al except.
llm_with_fallback = resilient_llm.with_fallbacks(
    [gemini_structured],
    exceptions_to_handle=TRANSPORT_ERRORS)

# --- PROMPT Y CADENA ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extraé la información estructurada del texto del usuario."),
    ("human", "{input}"),
])
chain = prompt | llm_with_fallback  # reintentos + fallback

sample_text = (
    "LangGraph es una librería de LangChain para construir agentes "
    "con estado. Me parece muy útil, aunque la documentación "
    "todavía tiene huecos.")

# --- EJECUCIÓN ---
async def main():
    try:
        result = await chain.ainvoke({"input": sample_text})
        print(result.model_dump_json(indent=2))
    # El modelo respondió pero no respetó el esquema
    except ValidationError as e:
        print("Error de validación:", type(e).__name__)
    # Fallaron OpenAI (con reintentos) Y Gemini, API key inválida, etc.
    except Exception as e:
        print("Fallaron todos los modelos:", type(e).__name__, "-", e)

asyncio.run(main())