# mytorch is your torch :fire:
A bag of tricks, aux functions to ease my pytorch experience.

Many parts here are inspired/copied from [fast.ai](https://github.com/fastai/fastai).
However, I've tried to keep is such that the control of model (model architecture), vocabulary, preprocessing is always maintained outside of this library.
The [training loop](./mytorch/loops.py), [data samplers](./mytorch/dataiters.py) etc can be used independent of anything else in here, but ofcourse work better together.

I'll be adding proper documentation, examples here, gradually.


# Features

1. **Training Loop**
    - Callbacks @ epoch start and end
    - Weight Decay (see [this blog post](https://www.fast.ai/2018/07/02/adam-weight-decay/) )
    - :scissors: Gradient Clipping
    - :floppy_disk: Model Saving 
    - :bell: Mobile push notifications @ the end of training :ghost: ( [See Usage](#notifications)) )
    
2. **Sortist Sampling** 
    
    TODO: link & description & refs
3. **Custom Learning Rate Schedules** 

    TODO: link & description & refs
4. Customisability & Flat Hierarchy
    
    Use/Ignore most parts of the library. Will not hide code from you, and you retain control over your models. 

# Usage


## Simplest Use Case
```
import torch, torch.nn as nn, numpy as np

# Assuming that you have a torch model with a predict and a forward function.
# model = MyModel()
assert type(model) is nn.Module

# X, Y are input and output labels for a text classification task with four classes. 200 examples.
X_trn = np.random.randint(0, 100, (200, 4))
Y_trn = np.random.randint(0, 4, (200, 1))
X_val = np.random.randint(0, 100, (100, 4))
Y_val = np.random.randint(0, 4, (100, 1))

# Preparing data
data = {"train":{"x":X_trn, "y":Y_trn}, "val":{"x":X_val, "y":Y_val} }

# Specifying other hyperparameters
epochs = 10
optimizer = torch.optim.SGD(model.parameters())
loss_function = nn.functional.cross_entropy
train_function = model      # or model.forward
predict_function = model.predict

train_acc, valid_acc, train_loss = loops.simplest_loop(epochs=epochs, data=data, opt=optimizer,
                                                        loss_fn=loss_function, 
                                                        rain_fn=train_function,
                                                        predict_fn=predict_function)
```

## Slightly more complex examples

@TODO: They exist! Just need to add examples :sweat_smile:
1. Custom eval
2. Custom data sampler
3. Custom learning rate annealing schedules

## Saving the model
@TODO


## Notifications
The training loop can send notifications to your phone informing you that your model's done training and report metrics alongwith.
We use [push.techulus.com](https://push.techulus.com/) to do so and you'll need the app on your phone.
*If you're not bothered, this part of the code will stay out of your way.* 
But If you'd like this completely unnecessary gimmick, follow along:

1. Get the app. [Play Store](https://play.google.com/store/apps/details?id=com.techulus.push) |  [AppStore](https://itunes.apple.com/us/app/push-by-techulus/id1444391917?ls=1&mt=8)
2. Sign In/Up and get yout **api key**
3. Making the key available. Options:
    1. in a file, named `./push-techulus-key`, in plaintext at the root dir of this folder. You could just `echo 'your-api-key' >> ./push-techulus-ley`.
    2. through arguments to the training loop (_@TODO: Add link :to relevant line_) as a string
4. Pass flag to loop, to enable notifications
5. Done :balloon: You'll be notified when your model's done training.

# Upcoming
1. :bangbang: Tests
2. :exclamation: Packaging this up to easy set-up post cloning (its a hack right now. Sorry ^.^)
3. Using FastProgress for progress + live plotting
4. ?? (tell me [here](https://github.com/geraltofrivia/mytorch/issues))  

# Contributions
I'm eager to implement more tricks/features in the library, while maintaining the flat structure (and ensuring backward compatibility). 
Open to suggestions and contributions. Thanks! 
