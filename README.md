# Extracción de entidades con LangChain (Pydantic + reintentos + fallback)

Pipeline que usa `gpt-4o` con salida estructurada (Pydantic), reintentos ante
errores de red y Gemini como modelo de respaldo.

Pasos pensados para **Git Bash en Windows** y **Python 3.12**.

## Requisitos

- Python 3.12 instalado (`python3.12 --version` o `py -3.12 --version`)
- Git Bash
- Una API key de OpenAI (necesita saldo): https://platform.openai.com
- Una API key de Gemini: https://aistudio.google.com

## 1. Crear el entorno virtual

```bash
python3.12 -m venv venv
```

Si `python3.12` no funciona, probá con:

```bash
py -3.12 -m venv venv
```

## 2. Activar el entorno virtual

En Windows el script de activación está en `Scripts/` (no en `bin/`):

```bash
source venv/Scripts/activate
```

Si se activó bien, aparece `(venv)` al principio de la línea.

## 3. Instalar las dependencias

```bash
pip install langchain-openai langchain-google-genai python-dotenv
```

Esto instala también, como dependencias, `langchain-core`, `pydantic` y `openai`.

## 4. Crear el archivo `.env` con las claves

Reemplazá los valores por tus claves reales:

```bash
OPENAI_API_KEY=sk-tu-clave-de-openai
GOOGLE_API_KEY=tu-clave-de-gemini
```

Reglas del `.env`:
- Formato `CLAVE=valor`, sin espacios alrededor del `=` y sin comillas.
- El archivo debe llamarse exactamente `.env` (no `.env.txt`).

## 5. Crear el `.gitignore`

Para no subir las claves ni el entorno virtual a git:

```bash
.env
venv/

```

## 6. Ejecutar el script

```bash
python extraccion.py
```

(Cambiá `extraccion.py` por el nombre real de tu archivo.)

Salida esperada: un JSON con 3 campos. Los valores pueden variar.

```json
{
  "topic": "LangGraph",
  "entities": ["LangGraph", "LangChain"],
  "sentiment_score": 0.6
}
```

