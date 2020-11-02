from django.contrib import admin

# Register your models here.

from .models import VrpProblem, VrpPoint, RoutingPlan

admin.site.register(VrpProblem)
admin.site.register(VrpPoint)
admin.site.register(RoutingPlan)