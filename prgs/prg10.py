

from PIL import Image, ImageEnhance  
im = Image.open(r"flower.jpg")
im.show()
im3 = ImageEnhance.Color(im) 
im3.enhance(4.3).show()






