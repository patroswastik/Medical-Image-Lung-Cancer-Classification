from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from . import masking
from . import prediction_model

def home(request):
    if request.method != 'POST' :
        return render(request,'spa.html',{"status" : "notdone"})
    else:
        myfile = request.FILES["image_file"]
        Name_of_user = request.POST.get("user_name")
        #print ("############################  ",Name_of_user)
        fs=FileSystemStorage()
        filename=fs.save(myfile.name,myfile)
        uploaded_image_url=fs.url(filename)
        #print (uploaded_image_url)

        final_image_url=masking.pre_process_img(uploaded_image_url)
        
        #final_image_url= final_image_url.replace("B:\\major_project_2\\major_gui","")
        #print (final_image_url)
        if prediction_model.Nodule_prediction(final_image_url) == True:
            if prediction_model.Cancer_prediction(final_image_url) == True:
                result = "Cancer"
            else:
                result = "Not Cancer"
        else :
            result = "Non Nodule"            

        return render(request,'spa.html',{"prediction" : "yes" ,"name" : Name_of_user ,"result": result ,"status":"done"})    

