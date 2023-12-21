# Usar la imagen base especificada para entornos de desarrollo de Python
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar solo el archivo de requerimientos para aprovechar la caché de capas de Docker
COPY requirements.txt ./

# Instalar las dependencias de Python especificadas en el archivo de requerimientos
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos de la aplicación al directorio de trabajo del contenedor
COPY . .

# Exponer el puerto que Streamlit utiliza por defecto
EXPOSE 8501

# Definir el comando para ejecutar la aplicación Streamlit
CMD ["streamlit", "run", "streamlit_app.py"]
