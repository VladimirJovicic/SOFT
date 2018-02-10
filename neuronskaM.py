import os
import cv2
import fun_NM
import numpy as np


ocekivaniIzlazi = []
slikeBrojevaZaTreniranje = []
for filename in os.listdir('data_set_brojeva\\trening'):
    #print(filename)
    brojZaTreniranje = filename[0]
    img = cv2.imread('data_set_brojeva\\trening\\' + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    output = np.zeros(10)
    output[int(brojZaTreniranje)] = 1

    ocekivaniIzlazi.append(output)
    slikeBrojevaZaTreniranje.append(img)

inputs = fun_NM.prepare_for_ann(slikeBrojevaZaTreniranje)
ann = fun_NM.create_ann()
ann = fun_NM.train_ann(ann, inputs, ocekivaniIzlazi)

# save the model to disk
model_json = ann.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
ann.save_weights("model.h5")
print("Saved model to disk")