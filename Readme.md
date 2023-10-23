
# NEMO: Asistente virtual de turismo en México

## **Descripción**

El objetivo principal de este proyecto es facilitar el acceso a la información global sobre oportunidades de viaje de turismo y paquetes personalizados que permitan a un viajero que desea conocer este país tener alternativas al alcance de su economía y de su planificación de viaje.



## **Configuración**

El funcionamiento del proyecto se puede manipular mediante el archivo **`config.yaml`**. Para utilizar el proyecto desde cero, sigue estos pasos:

1. **Extracción de textos**: Ejecute **`nemo/text_extractor.py`**. Este script exportará a la carpeta **`data`** un archivo jsonl con todos los archivos markdown de las documentaciones indicadas en la variable **`github`** en el archivo **`config.yaml`**. Estos archivos serán limpiados por **`text_extractor.py`** y estarán listos para ser divididos en Documentos de Langchain.
2. **Recreación de la base de datos de Chroma**: Ajuste la variable **`recreate_chroma_db`** en **`config.yaml`** a **`true`**. Esto indica que se creará una nueva base de datos de vectores Chroma y se almacenará localmente con el nombre "chroma_docs".
3. **Incrustación y almacenamiento de documentos**: Ejecute **`nemo/ai_conversation.py`**. Este script cargará el archivo jsonl creado en el paso 1 (asegúrese de agregar su nombre al archivo **`config.yaml`** en la variable **`jsonl_database_path`**). Luego, recreará la base de datos de Chroma incrustando todos los archivos json en el archivo jsonl creado, dividiéndolos y almacenándolos en la base de datos de vectores de Chroma para crear un índice.
4. **Uso de la base de datos existente**: Una vez que la base de datos de Chroma ha sido recreada, no es necesario volver a hacerlo. En la configuración, la variable **`recreate_chroma_db`** puede ajustarse a **`false`**, de modo que se utilizará la base de datos de Chroma existente en lugar de crear una nueva que implique volver a incrustar todos los archivos.
5. **Modo de chat**: Ajuste la variable **`chat_type`** en **`config.yaml`** a **`qa_chat`** para una interacción en modo de preguntas y respuestas, o a **`memory_chat`** para un chat con memoria. En el modo de preguntas y respuestas, el chatbot genera respuestas basándose puramente en la consulta actual sin considerar el historial de la conversación. En el modo de chat con memoria, el chatbot puede recordar partes de la conversación para generar respuestas más contextualizadas.
6. **Interacción con los documentos**: Al ejecutar **`nemo/conversation_ai.py`**, podrás chatear con todos los documentos obtenidos de Github.

## Instalación de dependencias

### **Instalación con Poetry (recomendada)**

Puede instalar las dependencias y crear un entorno virtual utilizando Poetry con el siguiente comando:

``` shell
poetry install
```

Luego, puede ejecutar el proyecto con:

``` shell
poetry run python nemo/ai_conversation.py
```


### **Instalación con Pip**

Como alternativa, también puede instalar las dependencias utilizando Pip con el siguiente comando:

``` shell
pip install -r requirements.txt
```

Luego, puede ejecutar el proyecto con:

``` shell
python hashira/ai_conversation.py
```

### **Instalación con Conda**

Además, puede optar por utilizar Conda para la instalación. Primero, cree un nuevo entorno con Conda:

``` shell
conda create --name myenv
```

Luego, active el entorno:

``` shell
conda activate myenv
```

A continuación, instale las dependencias:

``` shell
conda install --file requirements.txt
```

Por último, ejecute el proyecto con:

``` shell
python nemo/ai_conversation.py
```

Recuerde reemplazar "myenv" con el nombre que desee para su entorno.