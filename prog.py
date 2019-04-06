a = "data:image/jpeg;base64,/9j/4RYkRXhpZgAAT"

# program to check whether it is jpg or jpeg or png
if(a[11]=='j' and a[12]=='p' and a[13]=='e' and a[14]=='g'):
    print('jpeg')
elif(a[11]=='j' and a[12]=='p' and a[13]=='g'):
    print('jpg')
elif(a[11]=='p' and a[12]=='n' and a[13]=='g'):
    print('png')
else:
    print("Invalid format")