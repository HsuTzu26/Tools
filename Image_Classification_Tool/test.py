import cv2
import matplotlib.pyplot as plt

# path = 'C:\\CYLab\\HSU\\HandOK_non_OK\\1.png'
foot_path = "C:\\CYLab_Hsuan\\Hand\\MTSS_JPG_NoWL\\IMG679_JSN.jpg"
img = cv2.imread(foot_path)

plt.imshow(img)
plt.show()
# cv2.imshow('test',img)
# cv2.waitKey(0)
# cv2.destroyWindows()
