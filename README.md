# sk-graph-estimator
SKGraphEstimator is a neural network builder that combines scikit-learn's user-friendly features with Keras's versatility. You can create very complex networks easily, allowing you to test vastly different architectures quickly and efficiently. Further, as it is a subclass of scikit-learn's BaseEstimator, it can be used in things like GridSearchCV.

# Example code
```python
from sk_graph_estimator.estimator import SKGraphEstimator

model = SKGraphEstimator(model_structure=[
    {
        'type':'I',
        'branches':[
            [
                {'type':'C','filters':32,
                 'kernel_size':(1,1),
                 'activation':'relu'}
            ],

            [
                {'type':'C','filters':32,
                 'kernel_size':(3,3),
                 'padding':'same',
                 'activation':'relu'}
            ],

            [
                {'type':'C','filters':32,
                 'kernel_size':(5,5),
                 'padding':'same',
                 'activation':'relu'}
            ]
        ]
    },

    {'type':'GAP'},
    {'type':'D','units':1,'activation':'linear'}
]
,epochs=20,learning_rate=5e-3,random_state=42)
```

# Why I built this
Often, when performing research, it's useful to test different machine learning models. Whether that be through tweaking the hyperparameters or changing your framework, it was interesting to see how different models performed.

However, when trying to work with neural networks, I encountered a block. I either had to use sklearn's MLPRegressor, which significantly limits your ability, or attempt to work with KerasRegressor from scikeras, which still requires you to rewrite dozens of lines of code when attempting to use a different model architecture. Debugging was a chore, and for more complex networks like Inception, ResNet, and Xception, it was a pain to code and read.

This is why I created SKGraphEstimator. It allows you to easily specify complex model structures with something resembling a flow chart, making it significantly more user-friendly while maintaining most of Keras's versatility. You can create complex and more readable just by coding something like:
```python
[
    {'type':'C', 'filters':32,
     'kernel_size':(3,3),
     'padding':'same',
     'activation':'relu'},

    {
        'type':'R',
        'layers':[
            {'type':'C','filters':32,
             'kernel_size':(3,3),
             'padding':'same',
             'activation':'relu'},

            {'type':'C','filters':32,
             'kernel_size':(3,3),
             'padding':'same',
             'activation':'linear'}
        ],
        'final_activation':'relu'
    },

    {'type':'GAP'},
    {'type':'D','units':1,'activation':'linear'}
]
```
Then, with many model structures created already, you can run it through a GridSearchCV or add it to your code alongside any other sklearn regressor, and it will behave just fine.