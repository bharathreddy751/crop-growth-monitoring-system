from django.urls import path,include
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    # 1. Login - Corrected template_name usage
    path('accounts/login/', auth_views.LoginView.as_view(template_name='accounts/login.html'), name='login'),
    
    # 2. Signup - Removed the repeated 'name' argument and fixed the path
    path('accounts/signup/', views.signup_view, name='signup'),
    
    # 3. Logout - Correctly added template_name as a keyword argument inside as_view()
    path('accounts/logout/', auth_views.LogoutView.as_view(template_name='accounts/logout.html'), name='logout'),
    
    # 4. Rest of the routes
    path('analysis/', views.analysis, name='analysis'),

    path('accounts/', include('allauth.urls')),
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('weather/', views.get_weather, name='get_weather'),
    path('treatment-report/', views.treatment_report, name='treatment_report'),
    path('store-location/', views.store_location, name='store_location'),
    path('store-analysis/', views.store_analysis, name='store_analysis'),

]