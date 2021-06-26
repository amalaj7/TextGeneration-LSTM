import numpy
from keras.models import load_model

model = load_model('textmodel.h5')
print(model.summary())

