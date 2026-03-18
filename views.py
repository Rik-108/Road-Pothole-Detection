#Views.py
from django.shortcuts import render

from django.http import HttpResponseRedirect
from django.urls import reverse_lazy
from django.views.generic import TemplateView
from employee.forms import EmployeeForm

from django.views.generic import DetailView
from employee.models import Employee

class EmployeeImage(TemplateView):
    form = EmployeeForm
    template_name = 'emp_image.html'

    def post(self, request, *args, **kwargs):
        form = EmployeeForm(request.POST, request.FILES)

        if form.is_valid():
            obj = form.save()

            return HttpResponseRedirect(reverse_lazy('emp_image_display', kwargs={'pk': obj.id}))

        context = self.get_context_data(form=form)
        return self.render_to_response(context)

    def get(self, request, *args, **kwargs):
        return self.post(request, *args, **kwargs)

class EmpImageDisplay(DetailView):
    model = Employee
    template_name = 'emp_image_display.html'
    context_object_name = 'emp'


def Road Pothole(request):
    result1 = Employee.objects.latest('id')
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import h5py
    models = keras.models.load_model('C:/Users/SPIRO73-PYTHON/Desktop/smb/Deep_Learning/Road Pothole Classification/Deploy/employee/e.h5')
    from tensorflow.keras.preprocessing import image
    test_image = image.load_img('C:/Users/SPIRO73-PYTHON/Desktop/smb/Deep_Learning/Road Pothole Classification/Deploy/media/' + str(result1), target_size=(225, 225))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = models.predict(test_image)
    prediction = result[0]
    prediction = list(prediction)
classes=['plain', 'pothole']
    output = zip(classes, prediction)
    output = dict(output)
if output['plain']==1.0:
    print("plain")
elif output['pothole']==1.0:
    print("pothole")
    return render(request, "result.html", {"out": a})



#emp_image.html
<!DOCTYPE html> 
<html lang="en"> 
<head> 
    <meta charset="UTF-8"> 
    <title>image upload example</title> 
</head>
<style>
  label
{
font-size: 20px;
color:purple;
font-family:  Algerian;
}
body
  {
    background: url(../static/image/emo.png);
    background-repeat: no-repeat;
    background-position: center;
    background-size: 100% 100%;
    min-height: 100vh;
}
<!--  button-->
<!--{-->
<!--font-size: 20px;-->
<!--font-family:  wide latin;-->
<!--color:black;-->
<!--background-color: green;-->
<!--  box-shadow: 5px 5px blue, 10px 10px red, 15px 15px green;-->
<!--}-->
.button {
  background-color: #4CAF50; /* Green */
  border: none;
  color: white;
  padding: 16px 32px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  margin: 4px 2px;
  transition-duration: 0.4s;
  cursor: pointer;
}
.button1 {
  background-color: white;
  color: black;
  border: 2px solid #4CAF50;
}
#ss
{
font-size: 20px;
color:#20fc03;
background-color: black;
font-family:  Times new roman;
}
.button1:hover {
  background-color: #4CAF50;
  color: white;
}

h2
{
font-size: 20px;
color:black;
background-color: white;
font-family:  Times new roman;
}
a
{
font-size: 20px;
color:yellow;
font-family:  Times new roman;
}
.alerts-border {
    border: 4px #ff0000 dashed;

    animation: blink 0.2s;
    animation-iteration-count: infinite;
}

@keyframes blink { 50% { border-color:yellow ; }
}
button
{
font-family: Algerian;
font-size: 35px;
font-weight: bold;
}
.alerts-border
{
width:500px;
margin-top:7%;
margin-left:28%;
padding: 10px 10px 10px 10px;
</style>
<body style="background-color: lightblue;">
<center><h2 class="blink_me">ROAD POTHOLE CLASSIFICATION USING DEEP LEARNING ALGORITHM</h2></center><br><br><br><br>
<div class=" alerts-border"style="  box-shadow: 0 19px 38px rgba(0,0,0,0.30), 0 15px 12px rgba(0,0,0,0.22);">
    <div id="aa" class="card-header"><h2>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;UPLOAD IMAGE HERE</h2></div>
  <div class="card-body">
      <div class="login-form">
    <form method = "post" enctype="multipart/form-data"> 
        {% csrf_token %} 
        {{ form.as_p }} 
        <button class="button button1" type="submit">Upload</button>
    </form>
      </div>
  </div>
</div>
</body> 
</html> 

#emp_image_display.html
<!DOCTYPE html>
<html>
<style>
  label
{
font-size: 20px;
color:red;
font-family:  Algerian;
}
body
  {
    background: url(../static/image/emo.gif));
    background-repeat: no-repeat;
    background-position: center;
    background-size: cover;
    -webkit-background-size: cover;
    -moz-background-size: cover;
    -o-background-size: cover;
    min-height: 100vh;
}
  button
{
font-size: 20px;
font-family:  wide latin;
color:black;
background-color: green;
  box-shadow: 5px 5px blue, 10px 10px red, 15px 15px green;
}
#ss
{
font-size: 20px;
color:#20fc03;
background-color: black;
font-family:  Times new roman;
}
h2
{
font-size: 20px;
color:black;
background-color: white;
font-family:  Times new roman;
}
a
{
font-size: 20px;
color:#FA2204 ;
font-family:  Times new roman;
}
.alerts-border {
    border: 4px #ff0000 dashed;

    animation: blink 0.2s;
    animation-iteration-count: infinite;
}
@keyframes blink { 50% { border-color:yellow ; }
}
</style>
<body>
{% load static %}
<center>
    <img src="{{emp.UPLOAD_ROAD POTHOLE_IMAGE.url}}" alt="Smiley face" width="225" height="225">
    <br>
    <a href="{% url 'Road Pothole' %}">Result</a>&#160;&#160;&#160;
    <a href="{% url 'home' %}">Go Back!!!</a>
</center>
</body>
</html>

#result.html
{% load static %}
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ROAD POTHOLE CLASSIFICATION OUTPUT RESULT</title>
</head>
<style>
  label
{
font-size: 20px;
color:red;
font-family:  Algerian;
}
body
  {
    background: url(../static/image/emo1.png);
    background-repeat: no-repeat;
    background-position: center;
    background-size: cover;
    -webkit-background-size: cover;
    -moz-background-size: cover;
    -o-background-size: cover;
    min-height: 100vh;
}
  button
{
font-size: 20px;
font-family:  wide latin;
color:black;
background-color: green;
  box-shadow: 5px 5px blue, 10px 10px red, 15px 15px green;
}
#ss
{
font-size: 20px;
color:#20fc03;
background-color: black;
font-family:  Times new roman;
}
.blink_me {
  animation: blinker 2s linear infinite;
}
@keyframes blinker {
  50% {
    opacity: 0;
  }
}
h2
{
background-color: #101010;
font-family: Algerian;
font-size: 33px;
letter-spacing: 3px;
color:#56ff00;
}
h1,h3
{
font-family: Algerian;
font-size: 33px;
letter-spacing: 3px;
color:red;
}
a
{
font-size: 20px;
color:black  ;
font-family:  Times new roman;
}
</style>
<body>
<center><h2 class="blink_me">ROAD POTHOLE CLASSIFICATION USING ARTIFICIAL NEURAL NETWORK</h2></center><br><br><br><br>
 <marquee direction="down"><center><b><h1>{{out}}</h1></b></center></marquee>

<h3><a href="{% url 'home' %}">Go Back!!!</a></h3>

</body>
</html>
