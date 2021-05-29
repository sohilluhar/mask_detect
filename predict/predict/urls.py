"""predict URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static

from . import view

urlpatterns = [
    path('admin/', admin.site.urls),

    url(r'^login$', view.login, name="login "),
    url(r'^verifyuser$', view.verify, name="verify_user"),
    url(r'^map$', view.map, name="map"),

    url(r'^home$', view.home, name="home"),
    url(r'^prediction$', view.prediction, name="prediction"),
    url(r'^upload$', view.upload, name="upload"),

    url(r'^chart$', view.chart, name="chart"),
    url(r'^field-distributions$', view.field_distributions, name="field-distributions"),
    url(r'^pneumonia-cases$', view.pneumonia_cases, name="pneumonia-cases"),
    url(r'^pixel$', view.poixel_analysis, name="pixel"),

    url(r'^news$', view.news, name="news"),
    url(r'^test$', view.test, name="test"),

    path('', view.home),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
