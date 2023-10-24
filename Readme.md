
# NEMO: Asistente virtual de recomendación turística para viajes a México

## **Descripción**

Proyecto en etapa de prototipo inicial, desarrollado para implementar un modelo de Inteligencia Artificial basado en el uso de Procesamiento de Lenguaje Natural empleando el uso de un modelo grande de lenguaje (LLM) y el framework Langchain para poder sacar el máximo potencial a estos modelos con la personalización específica para este chatbot en el uso de recomendaciones turísticas en México permitiendo a un viajero que desea conocer este país tener alternativas al alcance de su economía y de su planificación de viaje.


## Instalación de dependencias

### **Instalación con Poetry**

Puede instalar las dependencias y crear un entorno virtual utilizando Poetry con el siguiente comando:

``` shell
poetry install
```

### **Instalación con Pip**

Como alternativa, también puede instalar las dependencias utilizando Pip con el siguiente comando:

``` shell
pip install -r requirements.txt
```

## **Configuración**

El funcionamiento del proyecto inicialmente esta configurado para ejecutar desde una instancia local. Para utilizar el proyecto desde cero, sigue estos pasos:

1. **Clonación de proyecto**: Clone el proyecto de este repositorio para tener todos los archivos necesarios para correr el chatbot.
2. **Creación de entorno virtual (venv)**: Se recomienda crear una instancia virtual de Python para la instalación de dependencias aplicadas solamente para este proyecto en específico.
3. **Creación de carpeta model**: Para poder aplicar el modelo GGML que es una biblioteca de tensores diseñada para el aprendizaje automático para modelos grandes empleada por `Llama`, aplicando una herramienta open source para este proyecto se debe descargar esta versión aplicada en este proyecto y colocarlo en la carpeta **model**:``https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGML/blob/main/codellama-7b-instruct.ggmlv3.Q4_0.bin.``

### **Ejecución*

Por último, ejecute el proyecto asegurándose que se encuentra en la raíz del mismo con:

``` shell
python run_ai_conversation.py
```

