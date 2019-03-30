# -*- coding: utf-8 -*-

#todo 绘制训练过程中的损失曲线和精度曲线

import matplotlib.pyplot as plt
from zstar_toolclassify import train

#训练模型
history = train.train()
acc = history.history['acc']
loss = history.history['loss']

val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label = 'val_acc')
plt.title('accurancy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='val_loss')
plt.title('loss')
plt.legend()

plt.show()

