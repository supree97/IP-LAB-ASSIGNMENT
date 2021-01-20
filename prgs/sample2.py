from PIL import Image , ImageEnhance
im = Image.open(r"nature.jpg")
im.show()
im3 = ImageEnhance.Color(im)
im3.enhance(4.3).show()







