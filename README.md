# sitio-SIMA
 sitio proyecto SIMA

Para poder ejecutar una instancia local es necesario clonar este repositorio.

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
Nota: el texto que va despues del parametri -it puede variar de acuerdo al nombre del contenedor asignado, si desea conocer este nombre o el id del contenedor debe ejecutar.
```
docker container ls
```
Ya que se ejecuta la terminal se ejecuta
```
python manage.py makemigrations
python manage.py migrate
```
Después de esto será necesario crear una cuenta de super usuario para el acceso y la administración del sitio
```
python manage.py createsuperuser
```
Una vez realizadas estas acciones debería de poder visualizar el sitio a través de http://localhost y acceder con las credenciales que ha definido.

SBN.