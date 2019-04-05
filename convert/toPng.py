from PIL import Image
import base64
# from where we will get txt of image
nameRead = "test22222.txt"

# where we will write image
nameWrite = 'some_image.jpeg'

# format jpeg or png
fmt = 'jpeg'

with open(nameRead,"r") as r:
	imgstring = r.read()
imgstring = imgstring.partition(",")[2]
imgstring = base64.b64decode(imgstring)
with open(nameWrite, 'wb') as f:
    f.write(imgstring)