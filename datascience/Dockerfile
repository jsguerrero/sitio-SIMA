FROM python:3
ENV PYTHONUNBUFFERED 1

RUN mkdir /app
WORKDIR /app

COPY manage.py /app/
COPY requirements.txt /app/

RUN pip install -r requirements.txt
COPY . /app/

# collect static files
RUN DATABASE_URL='' python manage.py collectstatic --noinput

CMD python manage.py runserver 0.0.0.0:$PORT