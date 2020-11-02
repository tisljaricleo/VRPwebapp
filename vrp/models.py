from django.db import models

# Create your models here.

class VrpProblem(models.Model):
    name = models.CharField(max_length=100)
    depot_id = models.IntegerField()
    description = models.CharField(max_length=250)
    data_path = models.CharField(max_length=250, default="None")
    def __str__(self):
        return self.name

class VrpPoint(models.Model):
    problem = models.ForeignKey(VrpProblem, on_delete=models.CASCADE)
    lat = models.DecimalField(max_digits=30, decimal_places=26)
    lon = models.DecimalField(max_digits=30, decimal_places=26)    
    poind_id = models.IntegerField()
    def __str__(self):
        return "Problem id: " + str(self.problem) + " " + str(self.poind_id) 

class RoutingPlan(models.Model):
    problem = models.ForeignKey(VrpProblem, on_delete=models.CASCADE)
    vehicle_id = models.IntegerField()
    routing_plan = models.TextField()
    total_distance = models.IntegerField()
    def __str__(self):
        return "Problem: " + str(self.problem) 



