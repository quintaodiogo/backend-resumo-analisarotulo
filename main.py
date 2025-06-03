from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance
import pytesseract
import io
import os
import platform
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# Carrega .env com chave da OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ajusta o caminho do tesseract se for Windows
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

# Permite frontend separado (ajuste allow_origins se necessário)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Abre a imagem enviada
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Pré-processamento: escala de cinza e aumento de contraste
        image = image.convert("L")
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        # OCR
        try:
            ocr_text = pytesseract.image_to_string(image, lang='por', config="--psm 6")
        except:
            ocr_text = pytesseract.image_to_string(image, config="--psm 6")

        print("🧾 OCR extraído:\n", ocr_text)

        # Prompt para IA
        prompt = f"""
Você receberá o texto extraído por OCR de um rótulo de produto alimentício.

Sua tarefa é montar um JSON com os dados **somente se estiverem claramente presentes** no texto.  
Caso não consiga identificar uma informação, **deixe o campo em branco ou omita**.  
Não invente, não deduza, e não use conhecimento externo.

⚠️ **Sobre os ingredientes:**
- Liste apenas ingredientes visíveis no texto.
- Cada ingrediente deve ser um item separado no array `"ingredients"`.
- O campo `"name"` deve começar com letra maiúscula, mesmo que no OCR apareça em minúsculo.
- O campo `"description"` deve explicar a função, se puder ser inferida com clareza.
- O campo `"safe"` deve ser `true`, a menos que haja **indicação clara** de risco.

⚠️ **Sobre os dados nutricionais:**
- Copie exatamente como estiver no texto.
- Exemplo: `"Valor energético": "130 kcal"`, `"Categoria": "por porção"`.

⚠️ **Seções adicionais:**
- `"claims"` → Frases de marketing como "Fonte de cálcio", "Zero lactose".
- `"warnings"` → Avisos como "Contém leite", "Pode conter soja".
- `"servingSize"` → Porções.
- `"storageInstructions"` → Instruções de conservação.

📌 **Exemplo de estrutura esperada:**

{{
  "productName": "",
  "brand": "",
  "ingredients": [
    {{
      "name": "Leite Integral",
      "description": "Ingrediente principal, rico em cálcio e proteína.",
      "safe": true
    }}
  ],
  "nutrition": [
    {{
      "label": "Valor energético",
      "value": "130 kcal",
      "category": "por porção"
    }}
  ],
  "additionalInfo": {{
    "claims": ["Fonte de cálcio"],
    "warnings": ["Contém lactose"],
    "servingSize": "200 ml",
    "storageInstructions": "Manter refrigerado após aberto"
  }}
}}

📄 **Texto OCR a ser interpretado:**

\"\"\"{ocr_text}\"\"\"
"""




        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Você transforma rótulos de leite em JSON estruturado no formato solicitado."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content
        print("📦 JSON gerado:\n", content)

        # Limpa markdown e converte para JSON real
        match = re.search(r"```json\s*(.*)\s*```", content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r"```(.*)```", content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = content

        try:
            json_result = json.loads(json_str)
        except Exception:
            json_result = {"erro": "Não foi possível converter para JSON", "original": json_str}

        # Salva JSON limpo
        with open("ultimo_resultado.json", "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)

        return {"json_result": json_result}

    except Exception as e:
        return {"error": str(e)}

@app.get("/resultado")
def get_resultado():
    try:
        if os.path.exists("ultimo_resultado.json"):
            with open("ultimo_resultado.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"json_result": data}
        else:
            return {"mensagem": "Nenhum resultado gerado ainda."}
    except Exception as e:
        return {"erro": str(e)}
