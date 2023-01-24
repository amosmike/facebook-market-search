#%%
import PIL
from Image_Classifier import ProductImageCategoryDataset

dataset = ProductImageCategoryDataset()

# for i in range(10):

#     image, label = dataset[i]
#     image.show()
#     print(dataset.decoder[label])
image, label, description = dataset[7155]
image.show()
print(dataset.decoder[label])
print(description)
# print(dataset.decoder)