FROM python:3.7

# VARIABLES DE ENTORNO
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# INSTALAR COMANDO NETCAT
RUN apt-get update
RUN apt-get install -y netcat

# DEPENDENCIAS DE LA APLICACION
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# SE COPIA EL PROYECTO PARA EJECUTAR LA APLICACION
COPY . .

# EJECUCION DE COMANDOS AL INICIO DEL SISTEMA
# ENTRYPOINT ["/app/entrypoint.sh"]

# # collect static files
# RUN DATABASE_URL='' python manage.py collectstatic --noinput

CMD python manage.py runserver 0.0.0.0:$PORT