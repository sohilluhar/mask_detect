def predict(imgpath):
    from tensorflow.keras.preprocessing import image
    import os
    import numpy as np
    import pandas as pd
    import random
    import cv2
    import matplotlib.pyplot as plt
    from keras.models import Model, Sequential

    loaded_model = load_model('xray_model2.h5')
    img = plt.imread(imgpath)
    img = cv2.resize(img, (img_dims, img_dims))
    img = np.dstack([img, img, img])
    img = img.astype('float32') / 255

    img = np.array([img])

    preds = loaded_model.predict(img)
    return preds
