import os
import sys
import jsonlines
from langchain.schema import Document
from dotenv import load_dotenv

class DocsJSONLLoader:
    """
    Cargador de documentos de documentaciones en formato JSONL.

    Args:
        file_path (str): Ruta al archivo JSONL a cargar.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        """
        Carga los documentos de la ruta del archivo especificada durante la inicialización.

        Returns:
            Una lista de objetos Document.
        """
        with jsonlines.open(self.file_path) as reader:
            documents = []
            for obj in reader:
                page_content = obj.get("text", "")
                metadata = {
                    "title": obj.get("title", ""),
                    "repo_owner": obj.get("repo_owner", ""),
                    "repo_name": obj.get("repo_name", ""),
                }
                documents.append(Document(page_content=page_content, metadata=metadata))
        return documents


def get_openai_api_key():
    """
    Obtiene la clave API de OpenAI del entorno. Si no está disponible, detiene la ejecución del programa.

    Returns:
        La clave API de OpenAI.
    """
    # HUGGINGFACEHUB_API_TOKEN = "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"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Por favor crea una variable de ambiente OPENAI_API_KEY.")
        sys.exit()
    return openai_api_key


def get_huggingface_api_key():
    """
    Obtiene la clave API de HuggingFace del entorno. Si no está disponible, solicita al usuario que la ingrese.

    Returns:
        La clave API de HuggingFace.
    """
    load_dotenv()
    huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if not huggingface_api_key:
        huggingface_api_key = input("Por favor ingresa tu COHERE_API_KEY: ")
    return huggingface_api_key


def get_query_from_user() -> str:
    """
    Solicita una consulta al usuario.

    Returns:
        La consulta ingresada por el usuario.
    """
    try:
        query = input()
        return query
    except EOFError:
        print("Error: Input no esperado. Por favor intenta de nuevo.")
        return get_query_from_user()


def create_dir(path: str) -> None:
    """
    Crea un directorio si no existe.

    Args:
        path (str): Ruta del directorio a crear.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def remove_existing_file(file_path: str) -> None:
    """
    Elimina un archivo si existe.

    Args:
        file_path (str): Ruta del archivo a eliminar.
    """
    if os.path.exists(file_path):
        os.remove(file_path)