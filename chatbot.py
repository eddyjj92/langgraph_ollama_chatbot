import os
from typing import Annotated, List
from typing_extensions import TypedDict  # Importación necesaria para TypedDict [[3]]
from langchain_ollama import ChatOllama
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
from langgraph.constants import START
from langgraph.graph import StateGraph, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from chroma_rag import VersatSarasolaDocumentStore, initial_documents
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar el modelo LLM (en este caso, Cloudflare Workers AI)
# llm = ChatOllama(
#     model="gemma3:1b",
# )

# Configurar el modelo LLM (en este caso, Cloudflare Workers AI)
llm = CloudflareWorkersAI(
    account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
    api_token=os.getenv("CLOUDFLARE_API_KEY"),
    model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
)

# Inicializar el almacén de documentos Chroma
versatChromaStore = VersatSarasolaDocumentStore(initial_documents)

# Mensaje del sistema
sys_msg = SystemMessage(
    content="Eres un chatbot llamado Versabot especializado en el sistema de gestión contable-financiero cubano Versat Sarasola."
            "Sigue las siguientes instrucciones: "
            "1. Presentate de forma elocuente."
            "2. No proporciones información falsa sino la posees. "
            "3. No hables de productos o servicios de terceros no relacionados. "
            "4. Proporciona respuestas breves y concisas de no mas de 50 palabras. "
            "5. Comunícate siempre en español. "
            "6. Si no conoces la respuesta a una pregunta, indica claramente que no tienes esa información en lugar de especular o inventar una respuesta. "
            "7. Mantén un tono profesional y objetivo en todas tus respuestas. "
            "8. Evita compartir opiniones personales o juicios de valor; limita tus respuestas a hechos y procedimientos comprobados. "
            "9. No repitas información que ya ha sido proporcionada anteriormente a menos que sea necesario para la claridad de la respuesta. "
            "10. Si la pregunta del usuario es ambigua o carece de suficiente contexto, solicita aclaraciones antes de proporcionar una respuesta."
            "11. Siempre termina con preguntas de retroalimentación."
            "12. Pon emojis relacionados al tema de conversacion."
)


# Definición del estado
class State(TypedDict):
    messages: Annotated[List[dict], add_messages]  # Usar `List` para mensajes [[3]]


# Función del asistente
def assistant(state: State):
    # Obtener el último mensaje del usuario
    user_message = state["messages"][-1]
    if isinstance(user_message, HumanMessage):  # Verificar que es un mensaje del usuario
        user_query = user_message.content  # Acceder al atributo .content del mensaje
    else:
        user_query = ""  # Si no hay mensaje del usuario, asignar una cadena vacía

    # Recuperar documentos relevantes
    relevant_docs = versatChromaStore.retrieve_documents(user_query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Construir el historial de mensajes
    history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
    )

    # Crear el prompt para el modelo
    prompt = (
        f"{sys_msg.content}\n\n"
        f"Historial de la conversación:\n{history}\n\n"  # Incluir el historial [[3]]
        f"Contexto relevante:\n{context}\n\n"
        f"Pregunta del usuario: {user_query}"
    )

    # Invocar el modelo LLM
    ai_response = llm.invoke([SystemMessage(content=prompt)])

    # Asegurarse de que la respuesta sea un objeto AIMessage
    if isinstance(ai_response, str):  # Si la respuesta es una cadena, envolverla en AIMessage
        ai_response = AIMessage(content=ai_response)

    # Devolver el nuevo estado con la respuesta del modelo
    return {"messages": state["messages"] + [ai_response]}


# Función para inicializar el chatbot
def initialize_chatbot():
    # Configurar el guardado en memoria
    memory = MemorySaver()

    # Crear el grafo de estados
    builder = StateGraph(State)
    builder.add_node("assistant", assistant)  # Agregar el nodo del asistente
    builder.add_edge(START, "assistant")  # Conectar el nodo inicial al asistente

    # Compilar el grafo
    return builder.compile(checkpointer=memory)


# if __name__ == "__main__":
#     chatbot = initialize_chatbot()
#     thread_id = "unique_thread_id"
#
#     while True:
#         user_input = input("Usuario: ")
#         if user_input.lower() in ["salir", "exit"]:
#             print("Versabot: ¡Hasta luego! 👋")
#             break
#
#         # Invocar el chatbot
#         response = chatbot.invoke(
#             {"messages": [HumanMessage(content=user_input)]},
#             config={"configurable": {"thread_id": thread_id}}
#         )
#
#         # Imprimir el estado actual
#         print("Estado actual:", response["messages"])
#         print(f"Versabot: {response['messages'][-1].content}")