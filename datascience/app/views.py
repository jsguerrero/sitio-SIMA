from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import redirect

# SI ACCEDE A LA PAGINA PRINCIPAL
# Y NO HA INICIADO SESION
# SE REDIRIGE A LA PAGINA DE INICIO DE SESION
# SI YA HA INICIADO SESION
# SE REDIRIGE A LA PAGINA DE INICIO
def index(request):
    if request.user.is_authenticated:
        return render(request,'app/index.html')
    else:
        return render(request,'app/login.html')

# UNA VEZ QUE SE INCIA SESION
# SE REDIRIGE A LA PAGINA DE INICIO
# SI NO SE HA PODIDO INICIAR SESION
# SE IMPRIME MENSAJE DE ERROR
from django.contrib.auth import authenticate, login as dj_login
from django.contrib import messages
def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            dj_login(request, user)
            next_url = request.POST['next']
            print(next_url)
            if not next_url:
                next_url = 'app:index'
            return redirect(next_url)
        else:
            messages.error(request,'Datos incorrectos')
            return redirect('app:login')
    else:
        return render(request,'app/login.html')

from django.contrib.auth import logout as dj_logout
def logout(request):
    dj_logout(request)
    return redirect('app:index')