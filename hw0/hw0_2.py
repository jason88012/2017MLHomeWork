from PIL import Image
srcImg = Image.open("westbrook.jpg")
srcPix = srcImg.load()
newImg = Image.new('RGB', (srcImg.size[0], srcImg.size[1]))

for w in range(srcImg.size[0]):
	for h in range(srcImg.size[1]):
		newR = int(srcPix[w, h][0] / 2)
		newG = int(srcPix[w, h][1] / 2)
		newB = int(srcPix[w, h][2] / 2)
		newImg.putpixel((w, h), (newR, newG, newB))

newImg.save("Q2.jpg")		