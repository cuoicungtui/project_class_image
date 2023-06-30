import os
folder = './image_test'
index =1
for filename in os.listdir(folder):
    if filename.lower().endswith(".png"):   
        old_path = folder+'/'+filename
        new_path = folder+'/'+str(index)+'.png'
        os.rename(old_path,new_path)
        index +=1