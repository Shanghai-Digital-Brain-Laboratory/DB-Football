# Model Definition in MAPPO
A model (e.g. [basic](../light_malib/model/gr_football/basic/))) of MAPPO has the following componenets:
1. Actor
2. Critic
3. FeatureEncoder
4. Backbone (if Actor and Critic shares some computation)

The model is supposed to be a module containing the above components or it can be a class if properly defined use some [import hacks](../light_malib/model/gr_football/__init__.py) now. 

Actor, Critic, Backbone should all inherit the `torch.nn.Module` class. In order to have correct behaviors when automatically distributed, all computation associated with their parameters should be included in their `forward` function. (This is because `torch.nn.parallel.DisitributedDataParallel` automatically handles distributed gradients only for the `forward` function. A possible way is to call everything using `forward` as an entry, for example, `forward(func_name,*args,**kwargs)` and calls actual functions inside `forward`.)