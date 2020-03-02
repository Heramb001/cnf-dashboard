
# 1. To load the model :

import keras
import mdn
from keras.models import Model, load_model

filename= 'AEMDNNP_model.h5'
output_dim= 25
num_mixes= 1
model = keras.models.load_model(filename, custom_objects={'MDN':mdn.MDN, 'loss_func':mdn.get_mixture_loss_func(output_dim,num_mixes)})



# 2. To predict: you need to load three scalers: AEmdnCrossScaler, AEmdnParScaler, and pcaAEmdnCrossScaler then you do:

noise = x_train[s] * (1 + 0.05 * np.random.randn(1000, 202))
noise= sc.transform(noise) # sc is AEmdnCrossScaler
noise= pca.transform(noise) # pca is pcaAEmdnCrossScaler

pred= model.predict(noise)
pred=pred[1]
y_pred= np.apply_along_axis(mdn.sample_from_output,1,pred, 10, 1,temp=1.0, sigma_temp=0.05)
y_pred= y_pred[:,0,:]
y_pred= sc2.inverse_transform(y_pred) # sc2 is AEmdnParScaler