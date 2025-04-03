import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.cloudflare_workersai import CloudflareWorkersAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from dotenv import load_dotenv

class VersatSarasolaDocumentStore:
    def __init__(self, documents: List[Document], embeddings_provider: str = "cloudflare"):
        """
        Inicializa la clase con los documentos proporcionados y configura el almacén vectorial Chroma.

        :param documents: Lista de documentos a almacenar.
        :param embeddings_provider: Proveedor de embeddings a utilizar.
        """
        # Cargar variables de entorno
        load_dotenv()
        self.documents = documents
        self.embeddings_provider = embeddings_provider
        self.collection_name = "versat_sarasola_docs"
        self.persist_directory = "./chroma_db"
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self) -> Chroma:
        """
        Configura el almacén vectorial Chroma. Si el directorio persistente existe, carga la base de datos existente;
        de lo contrario, crea una nueva e ingesta los documentos proporcionados.

        :return: Instancia del almacén vectorial Chroma.
        """
        if self.embeddings_provider == "cloudflare":
            embeddings = CloudflareWorkersAIEmbeddings(
                model_name="@cf/baai/bge-large-en-v1.5",
                account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
                api_token=os.getenv("CLOUDFLARE_API_KEY"),
            )
        else:
            embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

        # Crear o cargar la base de datos
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            # Cargar la base de datos existente
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_function=embeddings
            )
        else:
            # Crear una nueva base de datos e ingestar documentos
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            doc_splits = text_splitter.split_documents(self.documents)
            vector_store = Chroma.from_documents(
                documents=doc_splits,
                collection_name=self.collection_name,
                embedding=embeddings,
                persist_directory=self.persist_directory,
            )
            # No es necesario llamar a vector_store.persist()

        return vector_store

    def retrieve_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Recupera los documentos más relevantes basados en la consulta proporcionada.

        :param query: Consulta de búsqueda.
        :param k: Número de documentos relevantes a recuperar.
        :return: Lista de documentos relevantes.
        """
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.5, 'k': k},

        )
        return retriever.invoke(query)


initial_documents = [
    Document(page_content="Versat Sarasola es un sistema de gestión contable-financiero desarrollado por la empresa de Soluciones Informáticas DATAZUCAR, perteneciente al Grupo Azucarero AZCUBA."),
    Document(page_content="El módulo de Nómina de Salarios en Versat Sarasola permite gestionar los pagos a empleados, facilitando el procesamiento de documentos primarios y la modelación de la actividad económico-financiera de la entidad."),
    Document(page_content="Versat Sarasola cuenta con múltiples módulos, entre ellos: Contabilidad General, Costos y Procesos, Finanzas (Caja y Banco), Inventarios, Activos Fijos, Facturación, Nómina de Salarios, Planificación, Configuración y Complementos."),
    Document(page_content="El módulo de Finanzas, Caja y Banco de Versat Sarasola se basa en el procesamiento de documentos primarios, permitiendo modelar la actividad económico-financiera de la entidad mediante conceptos de cobros y pagos definidos por el usuario."),
    Document(page_content="Versat Sarasola es reconocido como el primer sistema cubano de gestión contable-financiera autóctono y versátil, facilitando el trabajo con la dualidad de moneda y la gestión de presupuestos."),
    Document(page_content="La versión 2.10 de Versat Sarasola incorpora la opción de factura electrónica y firma digital, así como un módulo denominado Punto de Venta."),
    Document(page_content="Versat Sarasola es utilizado por más de 35,000 clientes activos en Cuba, incluyendo misiones médicas y embajadas en el exterior."),
    Document(page_content="El módulo de Planificación de Versat Sarasola permite cumplir con los requerimientos informativos de las diferentes etapas de elaboración de presupuestos, integrando la contabilidad y su ejecución bajo una misma concepción."),
    Document(page_content="Versat Sarasola es un sistema maduro, altamente configurable y de resultados comprobados tanto en el sector empresarial como en el presupuestado."),
    Document(page_content="El módulo de Inventarios en Versat Sarasola permite el control y registro detallado de los recursos materiales de la entidad, facilitando la gestión eficiente de los mismos."),
    Document(page_content="El módulo de Activos Fijos de Versat Sarasola ofrece herramientas para la gestión y control de los activos fijos de la entidad, incluyendo su depreciación y revalorización."),
    Document(page_content="El módulo de Facturación de Versat Sarasola facilita la emisión y control de facturas, integrándose con otros módulos para una gestión coherente de las operaciones comerciales."),
    Document(page_content="El módulo de Costos y Procesos de Versat Sarasola permite analizar y controlar los costos asociados a los procesos productivos y de servicios de la entidad."),
    Document(page_content="El módulo de Configuración de Versat Sarasola permite adaptar el sistema a las necesidades específicas de la entidad, definiendo parámetros y estructuras según los requerimientos del usuario."),
    Document(page_content="Versat Sarasola facilita trabajar en un entorno multiusuario, permitiendo la gestión simultánea de múltiples usuarios con diferentes roles y permisos."),
    Document(page_content="El sistema permite la generación de informes y análisis financieros en tiempo real, apoyando la toma de decisiones basada en datos actualizados."),
    Document(page_content="Versat Sarasola cuenta con una amplia documentación y soporte técnico, facilitando su implementación y uso en diversas entidades."),
    Document(page_content="El módulo de Complementos de Versat Sarasola ofrece funcionalidades adicionales que pueden ser integradas según las necesidades específicas de la entidad."),
    Document(page_content="Versat Sarasola está diseñado para ser empleado en cualquier tipo de entidad, adaptándose a las particularidades de diferentes sectores económicos."),
]