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
Você receberá o texto extraído de um rótulo de um produto alimentício (como leite, iogurte, bebida vegetal, etc.).  
Sua tarefa é organizar **somente as informações que aparecem explicitamente** nesse texto, no formato JSON abaixo.

⚠️ **Importante:**
- Não invente informações que não estão no texto.
- Não complete campos com base em suposições.
- Se algo não estiver claro ou estiver faltando, **deixe o campo em branco ou omita**.

🧠 **Sobre os ingredientes:**
- Liste **cada ingrediente separadamente**.
- O campo `"name"` deve conter o nome do ingrediente com a **primeira letra maiúscula**, mesmo que no texto OCR esteja em minúsculo.
- A `"description"` deve explicar, se possível, a função ou papel do ingrediente no produto.
- Se a descrição não puder ser determinada com segurança, pode deixar em branco.
- O campo `"safe"` deve ser definido como `true`, a menos que o ingrediente seja claramente identificado como perigoso ou alergênico.

🧪 **Sobre os dados nutricionais:**
- Extraia pares como: `"Valor energético": "130 kcal"`, `"Categoria": "por porção"`.
- Só inclua dados que realmente estiverem presentes no OCR.

📝 **Formato esperado:**

{{
  "productName": "",
  "brand": "",
  "ingredients": [
    {{
      "name": "",             // Sempre começar com letra maiúscula
      "description": "",
      "safe": true
    }}
  ],
  "nutrition": [
    {{
      "label": "",
      "value": "",
      "category": ""
    }}
  ],
  "additionalInfo": {{
    "claims": [],                  // Ex: "Fonte de cálcio", "Sem lactose"
    "warnings": [],                // Ex: "Contém derivados de leite"
    "servingSize": "",            // Ex: "200 ml"
    "storageInstructions": ""     // Ex: "Manter refrigerado após aberto"
  }}
}}

📌 **Exemplo válido de ingredients:**

"ingredients": [
  {{
    "name": "Leite integral",
    "description": "Ingrediente principal do produto, fonte de proteína e cálcio.",
    "safe": true
  }},
  {{
    "name": "Trifosfato pentassódico",
    "description": "Estabilizante utilizado para manter a textura do leite.",
    "safe": true
  }}
]

### Texto OCR:

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
