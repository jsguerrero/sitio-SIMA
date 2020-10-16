# sitio-SIMA
 sitio proyecto SIMA

Para poder ejecutar una instancia local por primera vez es necesario clonar este repositorio.

Tener instalado y corriendo Docker.

Entrar a la carpeta donde se descargo el repositorio, contruir la imagen y ejecutarla con el archivos docker compose.

```
docker-compose build
docker-compose up -d
```
Una vez que se esta ejecutando el contenedor, al iniciar por primera vez será necesario implementar las acciones necesarias de las aplicaciones Django además de crear un usuario administrador.
Para esto será necesario iniciar una consola dentro del contenedor que ejecuta la aplicación.
```
docker exec -it sitio-sima_web_1 /bin/bash
```
Nota: el texto que va despues del parametro -it puede variar de acuerdo al nombre del contenedor asignado, si desea conocer este nombre o el id del contenedor debe ejecutar.
```
docker container ls
```
será necesario aplicar los cambios de modelos de las aplicaciones Django, carcar los recursos estáticos del sitio y crear una cuenta de super usuario para el acceso y la administración. Ya que se tiene lista la terminal se ejecuta
```
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic
python manage.py createsuperuser
```
Una vez realizadas estas acciones se puede cerrar la termial de la imagen de Docker, con las teclas ctrl + d o la palabra exit. Ahora debería de poder visualizar el sitio a través de http://localhost y acceder con las credenciales que ha definido.

Para visualizar el dashboard del proyecto se debe acceder al menú  SIMA > Exploración de datos.

Una vez que ya no se trabajará con el sitio se recomienda detener las imagenes docker de la siguiente forma
```
docker-compose down
```

Despues de realizar estos pasos la primera vez, posteriormente solo será necesario ejecutar las imágenes de Docker dentro de la carpeta del proyecto.
```
docker-compose up -d
```

Y visualizar el dashboard web desde la dirección http://localhost. De igual forma una vez que ya no se trabajará con el sitio se recomienda detener las imagenes docker de la siguiente manera
```
docker-compose down
```

SBN.