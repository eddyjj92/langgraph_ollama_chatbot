import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from chatbot import initialize_chatbot, versatChromaStore
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

# Inicialización de la aplicación FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lista de orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todos los encabezados
)

react_graph_memory = initialize_chatbot()


# Modelo Pydantic para la solicitud
class QueryRequest(BaseModel):
    query: str
    session_id: str
    k: int = 3


# Ruta para recuperar documentos relevantes
@app.get("/retrieve_documents/{query}/{k}")
async def retrieve_documents(query: str, k: int):
    try:
        results = versatChromaStore.retrieve_documents(query, k)
        return {"documents": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        messages = react_graph_memory.invoke(
            {"messages": [HumanMessage(content=request.query)]},
            config={"configurable": {"thread_id": request.session_id}}
        )
        return {
            "response": messages["messages"][-1].content,
            "history": messages["messages"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class CustomStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)

        # Forzar el tipo MIME correcto para archivos JS
        if path.endswith(".js"):
            response.headers["Content-Type"] = "application/javascript"
        return response

# Ruta para servir el archivo index.html de la SPA
@app.get("/")
async def serve_spa():
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public/spa", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html no encontrado")

# Montaje del directorio de archivos estáticos
app.mount("/assets", CustomStaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "public/spa/assets")), name="assets")
# Ruta para servir la imagen bot.png
@app.get("/bot.png")
async def serve_bot_image():
    bot_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public/spa/", "bot.png")
    if os.path.exists(bot_image_path):
        return FileResponse(bot_image_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Imagen bot.png no encontrada")

# Ruta para servir la imagen user.png
@app.get("/user.png")
async def serve_user_image():
    user_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public/spa/", "user.png")
    if os.path.exists(user_image_path):
        return FileResponse(user_image_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Imagen user.png no encontrada")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
