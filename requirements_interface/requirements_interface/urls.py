"""
Root URL configuration for the Django project.

This module defines the URL routing for the entire project, including:
- Admin interface URLs
- Application-specific URL includes
- Static file serving in development mode
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("interface.urls", namespace="interface")),
    path("accounts/", include("accounts.urls", namespace="accounts")),
]

if settings.DEBUG is False:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
