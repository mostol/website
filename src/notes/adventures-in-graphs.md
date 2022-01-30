---
layout: post.njk
title: "Adventures in Graphs, Part 1: (Not) Everything's a Vision Problem"
tags: ["blog"]
---
<div class="cblock">

This last year, IARAI ran a [2021 Science4Cast competition](https://github.com/iarai/science4cast), with the task of turning research-soothsayer and predicting what topics will be linked together through future machine learning research publications (e.g. will "Deep Learning" and "Ordinary Differential Equations" show up in the same paper in the next three years?). I poked through the [example solution](https://github.com/iarai/science4cast/blob/main/Tutorial/tutorial.ipynb), which precomputed some metrics for each pair of topics in the training set and fed them to a simple three-layer neural network and thought, "I could beat that, no problem!"

Did I beat it? Yep! Was it "no problem"? Nope! In fact, I attempted two completely different approaches, and one of them didn't even beat the tutorial solution provided as a straightforward example. And that's the one I'll describe here, because even if it wasn't successful, it was an interesting process! (To me, at least.) Here's how it went.
</div>

## Sight setting
<div class="cblock">

Since the end goal of the competition is to be able to take in two topics and predict whether or not they will connect in the future, the most intuitive approach to me seemed to be feeding a representation of each topic into a neural network, and training that network to classify the outcome for us.

I know that there's a neuropsychologically-rooted understanding of deep neural networks, but I am not a neuropsychologist, so my best intuitive grasp of how neural nets work is rooted in the [Universal Approximation Theorem](https://wikipedia.org/wiki/Universal_approximation_theorem) and functions in their most basic form, which is "you put a thing in, and you get a thing out". If we can clearly define what we're putting in and what we want to get out, then we can set up a model that takes thing `A` in and spits thing `B` out and let our training algorithms optimize that process ✨*mathemagically*✨ to do it with data it hasn't seen before. Thinking about a deep learning model this way, while maye not the most rigorous or complex approach, has the added benefit of letting me focus on single idea, which helps me avoid getting bogged down in less-important implementation details or do something that's accidentally incompatible with the competition task.

The details in between are flexible, but here was my mantra for this approach:

1. The model must take take in representations of *two* nodes, and
2. The model must tell me, yes or no, *Will these two nodes be connected in 3 years?*

For the sake of flexibility, I decided that, while the competition expects me to predict results for three years from the present, that number shouldn't necessarily be baked in to the fundamental design of the model, so I dubbed this the "n-Years Model", where `n = 3` in this case. </div>

## Everything's a vision problem?
<div class="cblock">

So, how do we set up a model that takes in two node representations and spits out *the future*?

I could have sworn I remembered [Jeremy Howard](https://fast.ai) saying something along the lines of, "Almost anything can be treated as a computer vision problem." It turns out I can't find any record of that ever happening now, but there *is* a section highlighting how "Image Recognizers Can Tackle Non-Image Tasks" in the [first chapter of fast.ai's *fastbook*](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb). Either way, my first instinct was to convert the dataset into a computer vision problem by conjuring up some kind of image-based representation of the dataset, and then just using plain old image-classifying techniques to get it to make the right predictions. So the game plan is to feed the model an *image* representing two nodes, and have it classify whether or not they will be connected in three years.
</div>

### Step 1: Imagification
<div class="cblock">

The competition's tutorial solution relied on calculating the degree of each topic/node, as well as counting common neighbors shared with other nodes. This seemed like a reasonable approach, because with so little information, and no other context, each node is essentially *defined* by its neighbors—so it's not a bad idea to focus on those relationships.

Since each topic is semantically defined by its connections, can our image representation be pulled straight from a complete description of all of a node's relationships? Sure! For starters, we can convert our dataset into an [adjacency list](https://en.wikipedia.org/wiki/Adjacency_list), the sometimes less-popular but often more storage space-friendly sibling of the [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix).

An adjacency list actually lends itself really well to creating images from its information, because images are really just arrays or tensors, and in PyTorch ([and NumPy](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing), where PyTorch got the idea from), tensors have "fancy" indexing of arrays using other arrays. Why is that handy? Because we can theoretically take a list of non-adjacent node id's, use them as the indices to select from another existing array, and set them to whatever value(s) we choose. In other words, if we have a node's adjacency list, we can use that directly as an index to represent that node's neighbors as `1` (or some other value) in a tensor, while leaving all other values `0`.

```python
node0_neighbors = np.array([1, 4, 6, 7, 9]) # A made-up list of node 0's neighbor nodes.

# Create a "base" of zeros
base = torch.zeros(64000) # Each of the 64000 nodes gets a spot.

base[node0_neighbors] = 1 # Now, all of node 0's neighbors are `1`, and everything else is 0.
```

Getting a full-graph adjacency list to use for this isn't too hard, but there *is* one caveat: our list is going to need to retain information about not just node connectivity, but also time. Remember, our model is supposed to tell us whether two nodes will be connected in *n* years—so in order to provide the model with ground-truth data to train on, we need to know about nodes' connectivity three years from whatever we feed the model. We'll get to how we could incorporate that to use during training, but here's the creation of our adjacency list, making sure we retain the temporal info:

```python
# Using the built-in `defaultdict` from Python's `collections`
# lets us easily build a dict iteratively without having
# to know the final state its elements from the start:

from collections import defaultdict

adj_list = defaultdict(list)

# The `graph` is just an array listing all of
# the graph's edges, formatted as:
#   node1 | node2 | time

for edge in graph:
    n1,n2,t = edge

    adj_list[n1].append([n2,t])
    adj_list[n2].append([n1,t])

# To enable fancy indexing (and speed up other processes),
# we'll turn these lists into arrays by way of tensors. A 
# little hacky? Yes. But it was the fastest method I could find! 😅

for key,value in arr_dict.items():
    adj_list[k] = np.asarray(torch.LongTensor(value).T)
```

Now, if we want to create an "image" for a node's connectivity, we can use the little process from above and get this:
```python
# If there are ~64000 nodes, then 255 x 255 could be a big enough image size
image_shape = (255,255)

base = torch.zeros(np.prod(image_shape))

# Generate the image for node 37
edges = adj_list[37][0] # We have to index the 0th item: [0]: connected nodes, [1]: connection times
base[edges] = 1 # Set selected elements to 1

# We'll even make it look image-y by making it a square shape:
image = base.reshape(img_shape)
```

Neat! Now to deal with that *time* element. But wait…what's that? Something's derailing us—**\*A wild ambition appears!\*** 😮
</div>

### Tangent 1: *k*-hop neighbors
<div class="cblock">

You know what's (ostensibly) better than 1-hop neighborhoods at representing nodes and supplying substantial information to train on? *2-hop* neighborhoods! Right now, each of our images we feed to the model are poised to simply be basic representations of a node's immediate neighbors—but what if we also included the *neighbors of those neighbors* at a given point in time in our image? Wouldn't the sheer density and magnitude of additional relevant information—the unreasonable effectiveness of more data!—*blow the socks off* of the tutorial model?! 😱🤯 It turns out that it did not. But I went through the trouble of making 2nd order neighbors work, anyways, so here we are.

In theory, it shouldn't be too hard to get the second order neighbors for a node. We already have all of the neighbors in our adjacency list, so we can just index the adjacency list of each of a node's neighbors, and we're good! Here's a simple way to do that:

```python
src_node = 1 # Finding node 1's 2nd order neighbors

[adj_list[neighbor][0] for neighbor in adj_list[src_node][0]]
```

That's great, but we're also going to need to use the temporal data from all of these neighbors—as noted above, we need to restrict our view of the graph to certain time constraints, which means we can't just index *every* neighbor. Instead, we'll want something more like this:
```python
src_node = 1
t = 8000 # Set time cutoff to day 8000

# The third indexing selector on each of these only indexes
# the locations where the expression is `True`, i.e.
# wherever the time array is less than `t`.
[adj_list[neighbor][0][adj_list[neighbor][1] < t]
 for neighbor in
 adj_list[src_node][0][adj_list[src_node][1] < t]]
```

There is also a [much more convoluted way](#altered-matrix-multiplication) of doing this that entails implementing an altered matrix multiplication for a sparsely-described adjacency matrix, leveraging the fact that [the square of an adjacency matrix represents the number of walks of length two from one node to another](https://arxiv.org/abs/1207.3122). I originally opted for this approach out of fear that Python's iteration would be too slow with the method above for practical use during training, but after testing it out while writing this post…the above list comprehension is actually faster. Here's my final function to get a node's first and second order neighbors (with built-in functionality for getting up to *k* total layers of neighbors, although exponential list growth with this approach makes any *k* > 2 a bit problematic):

```python
def get_neighbors(adj_list, node, t=None, k=2):
    neighbors_list = []
    nbrs = [node]
    if t is None:
        for _ in range(k):
            # If our node *has* no neighbors, we'll get an error we need to handle.
            try:
                nbrs = np.concatenate([adj_list[node][0] for node in nbrs])
            except (IndexError, ValueError):
                nbrs = np.array([],dtype=np.int64)

            neighbors_list.append(nbrs)
    else:
        for _ in range(k):
            try:
                nbrs = np.concatenate([adj_list[node][0][adj_list[node][1] < t]
                                       for node in nbrs])
            except (IndexError, ValueError):
                nbrs = np.array([],dtype=np.int64)

            neighbors_list.append(nbrs)

    return neighbors_list
```

This returns a list of first order neighbors, followed by second neighbors (and up to *k*th order neighbors if we really wanted).
</div>

### Step 2: Temporal label-wrangling
<div class="cblock">

If you think about it, we don't really even need *any* information about the time that previous connections were made when we're doing predictions; it ought to be enough to know the connectivity of each node right now and whip up an image. But if we remember the second pillar of my model mantra:

> The model must tell me, yes or no, *Will these two nodes be connected in 3 years?*

in order to train the model, we're going to need to pass the model node representations and also *tell* it if those nodes happened to get connected within 3 years, so that it has enough context to actually answer the question.

In order to have lots of possible samples to train off of, it seems like a good idea to be able to pass the model *any* two nodes at *any* given point in time, and to get a yes/no as to whether they will be linked in three years. I found it easiest to think about this in terms of "do these nodes connect at any point at all?" and then handle the timing part appropriately after figuring that out—this way, we can maximize the number of actual connections we feed the model so that training is more balanced, instead of feeding it nodes-that-won't-connect 99.8% of the time. There are a few possibilities here:

1. One (or both) of the nodes has *no* connections at the given time. A blank slate is not particularly meaningful here, so it might be best to just avoid or ignore this case.
2. The nodes both have neighbors, but they *never* connect to one another. In this case, any time point is as good as another, so we can pick whichever one we want to feed the model. **BUT!** We have to be careful about the maximum timestamp we're willing to offer. If I feed the model two nodes as they exist on January 1, 2017, but my data only goes as far as 2019, then I know that they haven't been linked *yet*, but I *do not know* if they will connect within 3 years because it hasn't *been* three years yet. So in this case, we can input the nodes as they are at any date, *as long as* that date is beyond *n* years of our most up-to-date data.
3. The nodes do connect at some point. Keeping in mind that we want to maximize our already-scarce supply of positive samples, we probably want to make sure we pick a date that puts these cases to good use. If two topics are linked in March 2009, for example, I don't want to give the model their representations as they exist in April 2003, because based on the April 2003 images, the answer to the question, *Will these two nodes be connected in 3 years?* is `no`. This is why focusing on connection status over dates is handy—in these instances, we can make sure we pass representations that are within *n* years of connection into our model for training so we have as many positive samples as possible.

With all that forethought out of the way, we can dive into setting up and training the model!
</div>

## Transforms & training
<div class="cblock">

For training, I used fast.ai, leveraging its [mid-level API](https://docs.fast.ai/tutorial.siamese.html#Using-the-mid-level-API) to work nicely with the above conditions. I decided that, from a user standpoint, the most ideal and intuitive way to interact with the model would be to simply pass in two node id's and let the model either train on that info or predict on in. Here's how we can make that happen with fast.ai's [`Transform`](https://fastcore.fast.ai/transform#Transform)s.

A `Transform` is, at the most basic level, just a function: it takes an input and *transforms* it into something else. (You might say a `Transform` is a *function in disguise*.) They have some nifty bonus features—like type dispatch, potential reversability (to allow both encoding and *de*coding), and extensability to name a few—but my primary use here is to use them as a medium for creating fast.ai `DataBlock`s in a convenient format for my partucular data. There are quite a few ways to define a `Transform`, but I opted for simply extending the `Transform` class. That looks like this:

```python
class SomeKindOfTransform(Transform): pass
```

But for it do actually do anything, we need the `Transform` to have an `encodes` method; so a transform that simply squares its input might look something like this:

```python
class SquareTransform(Transform):
    def encodes(self, x):
        return x**2

SquareTransform(12) # Gives us 144.
```
</div>

### Defining `NYearsTransform`
<div class="cblock">

Alright! Now to define a `Transform` to accept two nodes and return an image representation of them based on their neighbors. First, we can to prepare all of the logic to handle the situations outlined above.
</div>

#### Initialization, contextual settings, utilities
<div class="cblock">

Our `Transform` is going to need some background information for every transformation it does; things like:

* What are the dimensions of the final image?
* What adjacency list are we pulling our information from?
* How big is the *n* in "*n* years," actually?

as well as other utilities that may be used repeatedly in the transformation process. We'll these up to be incorporated when we initialize a `NYearsTransform` object using the `__init__` function:

```python
class NYearsTransform(Transform):

    # Initialization
    def __init__(self, adj_list, mode, img_shape=None, n=3):
        self.adj_list = adj_list # Attach an inputted adjacency list to the object
        if mode in {'train','eval'}: #Specify a mode ("train" or "eval"). We'll get to this in a moment!
            self.mode = mode
        else:
            raise ValueError('`mode` should be `"train"` or `"eval"`')

        # Time-related attributes
        self.t_delta = n * 365 # Graph times are in days, so we need to have "n years" in terms of days


        ### Latest time on the graph. (Avoids checking nodes with no connections.)
        max_graph_time = max(edges[1].max() for edges in adj_list.values() if len(edges))

        ### For training, only edges formed `n` years before the latest data are "True" values for certain:
        self.t_cutoff = max_graph_time - self.t_delta

        # Image-making attributes
        ### If you manually define a shape, set that as the shape...
        if img_shape is not None:
            self.img_shape = shape 
        ### Otherwise, image is the smallest square possible based on number of nodes
        else:
            len_sqrt = int(max(adj_list.keys())**0.5)
            self.img_shape = (len_sqrt, len_sqrt)

```

One important operation we'll need every time we use our transform is converting our list of neighboring nodes to an image, as [mentioned previously](#something-something). We can convert the same approach into a function and add it as a built-in part of our class:

```python
class NYearsTransform(Transform):
    def __init__(self, adj_list, mode, img_shape=None, n=3):
        self.adj_list = adj_list
        if mode in {'train','eval'}:
            self.mode = mode
        else:
            raise ValueError('`mode` should be `"train"` or `"eval"`')

        self.t_delta = n * 365

        max_graph_time = max(edges[1].max() for edges in adj_list.values() if len(edges))
        self.t_cutoff = max_graph_time - self.t_delta

        if img_shape is not None:
            self.img_shape = shape 
        else:
            len_sqrt = int(max(adj_list.keys())**0.5)
            self.img_shape = (len_sqrt, len_sqrt)

    # *NEW* Adding a utility function to turn list of neighbors to image tensors.
    def get_img(self, edges):
        base = torch.zeros(np.prod(self.img_shape),dtype=torch.float)
        base[edges] = 1
        return TensorImage(base).reshape(self.img_shape)
```
</div>

#### Training vs. evaluation modes
<div class="cblock">

If you read my [earlier exposition](#Step-2) carefully, you may have caught wind of the subtle intimation that we're going to need to do fairly different things during training vs. inference. To deal with this I decide to set the `Transform` up to handle two disting "modes": `'train'`, where we carefully supply timeframes and contexts that are useful for training, and `'eval'`, where we just grab the most up-to-date connectivity information and go. I decided to just have the mode be determined by passing in a string. 

This is where we finally get to define the full behavior of the `Transform`! This is done with the associated `encodes` function. It will take in a pair of nodes (a `nodepair`) and, based on the mode, return the appropriate image. If we're in "Train" mode, the transform will return not just the image, but also the label correctly answering the question, *Will these two nodes be connected in *n* years?*

One bonus aspect we might want to consider is differentiating 1st and 2nd order neighbors in our image. It may be useful for the model to distinguish between those when trying to make a prediction. One way we could handle this is by having a (2 \* *k*)-channel image; in other words, each set of *k*th order neighbors gets its own layer for each node. But this might make visualization a bit tricky (if we want to actually look at the "images"), because most libraries expect an image to either be 1-channel or 3-channel, and it also means larger tensors are getting passed in. Another option would be summing layers together in a way that retains the seperability of the information at the end, which is the approach I opted for. For example, if we weight first order neighbors as `0.75` and second order neighbors as `0.25`, then we can tell that a node with `0` had no connections, a node with `0.75` was exclusively a first order neighbor, one with `0.25` is exclusively a second order neighbor, and `1.0` is a node that is connected by both one *and* two hops. Here's the final implementation of the `Transform` including summed-and-weighted neighbor layers and both transform modes.

```python
class NYearsTransform(Transform):

    def __init__(self, adj_list, mode, img_shape=None, n=3):
        self.adj_list = adj_list
        if mode in {'train','eval'}:
            self.mode = mode # Told you we'd get there!
        else:
            raise ValueError('`mode` should be `"train"` or `"eval"`')

        self.t_delta = n * 365

        max_graph_time = max(edges[1].max() for edges in adj_list.values() if len(edges))
        self.t_cutoff = max_graph_time - self.t_delta

        if img_shape is not None:
            self.img_shape = shape 
        else:
            len_sqrt = int(max(adj_list.keys())**0.5) + 1
            self.img_shape = (len_sqrt, len_sqrt)

        # *NEW* Adding in weighting for 1st and 2nd order neighbors.
        self.weights = (0.75, 0.25)

    # *NEW* Defining our encodes!
    def encodes(self, nodepair:Iterable):
        nodepair = {int(n) for n in nodepair}

        ### EVAL ### (Returns just a TensorImage)
        if self.mode == 'eval':
            # Create, weigh, and sum layers for each node
            img_layers = [sum(self.get_img(layer)*weight
                              for layer,weight
                              in zip(get_neighbors(self.adj_list,n),self.weights))
                          for n in nodepair]

            # Return stacked representation of nodes.
            return torch.stack(img_layers)

        ### TRAIN ### (Train mode will return the TensorImage *and* a label.)
        elif self.mode == 'train':
            # To check for connectivity, we need to look at nodes individually for a sec.
            nodes_copy = nodepair.copy()
            link_list = arr_dict[nodes_copy.pop()]
            partner_node = nodes_copy.pop()
            linked = partner_node in link_list[0] # A boolean value

            # If nodes connect
            if linked:
                lim = link_list[1][link_list[0] == partner_node] # Time of the connection

                # Passing that time into `get_neighbors` gets all connections *before* that point
                img_layers = [sum(self.get_img(layer)*weight
                                  for layer,weight
                                  in zip(get_neighbors(self.adj_list,n,lim),self.weights))
                              for n in nodepair]

                # If either edgelist is empty at this time (i.e. this is >= 1 node's first link ever)
                if any(layer.sum() == 0 for layer in img_layers):
                    label = 0

                # Otherwise (normal behavior):
                else:
                    partner_links = arr_dict[partner_node]

                    # Most recent link time before these two nodes linked:
                    maxlink = np.max([np.max(link_list[1][link_list[1] < lim]),
                                      np.max(partner_links[1][partner_links[1] < lim])])

                    # Verify previous link (image reference) and most recent link are actually within n years.
                    label = 1 if lim - maxlink < self.t_delta else 0

            # If nodes do not connect
            else:
                # Get connections as they are at `self.t_cutoff` 
                # (any later and we aren't *really* sure they won't connect in n years
                img_layers = [sum(self.get_img(layer)*weight
                                  for layer,weight
                                  in zip(get_neighbors(self.adj_list,n,self.t_cutoff),self.weights))
                              for n in nodepair]

                label = 0

            # Return image,label
            return torch.stack(img_layers),label

    def get_img(self, edges):
        base = torch.zeros(np.prod(self.img_shape),dtype=torch.float)
        base[edges] = 1
        return TensorImage(base).reshape(self.img_shape)
```

Now that our transform is ready to go, we can flesh out the training pipeline, and get training!.
</div>

### Training pipeline
<div class="cblock">

So we have a `Transform` that converts to topics into image representations of those topics based on their connectivity, and will also provide a label telling us whether those images represent two graphs that will be connected in *n* years. But how do we use that to create a model? All we have to do is pass in a bunch of node pairs to the transform, and then pass the output of the transform (the image and label) to a vision model, and train!

Fast.ai uses ["datablocks"](#link-to-datablock-api) for training—objects containing either the data and labels themselves, or a way to *get* the data and labels. While we don't have a built-in `DataBlock` that's fit to use with our current setup, we can make training and validation `DataBlock`s fairly simply from our `Transform`. The built-in `ImageBlock`, for example, takes in strings indicating image file paths and labels, uses those to get the actual images/labels, and passes those to the model—but doesn't actually store the images themselves *in* the datablock. We'll do something similar, but instead of getting images from a file, we're generating them based on simply inputting two node id's. Since we've built our own custom `Transform` that's not dependent upon an existing type of `DataBlock`, we'll use a [`TfmdList`](https://docs.fast.ai/data.core.html#TfmdLists) ("Transformed List") instead of an actual `DataBlock` object, but they're fundamentally the same: take in information and transform it on-the-fly into the format needed for training. But before we make a `TfmdList`, we'll need to select pairs of nodes we want to use for training and for validation.
</div>

#### Training and validation sets
<div class="cblock">

We could use a much larger training set, but to keep training time down, I'll start with something a bit more modest. Let's say…5000 samples? We could select 5000 random combinations of nodes for our training set, but remember that the dataset is *incredibly* sparse—so if we just pick node pairs as random, we'll end up with a *lot* of "these topics do not connect", so the point where training might not even accomplish anything and the model may just assume that the correct label is always 0. A better idea is to try to intentionally balance our training set so that it has roughly equal amounts of positive and negative samples; cases where we know for sure that the nodes will connect, along with cases where they won't. So let's split our training set up into equal "positive" and "negative" groups—the `TfmdList` will want these as lists, so we'll also make sure our sample node pairs are in that format:

```python
pos_size = 2500
neg_size = 2500

# Set up some random number generation in NumPy (with seed = 42)
rng = np.random.default_rng(42)

pos_size = 2500
neg_size = 2500

pos_sel = rng.choice(len(graph),pos_size,replace=False)
neg_sel = rng.choice(len(unconnected_v_pairs),neg_size,replace=False)

# Grab 2500 pairs that already exist from the graph (w/o time column)
pos_sample = list(graph[pos_sel,:2])

# Pull 2500 pairs from the `unconnected_v_list`, all of which are unconnected at this point
neg_sample = list(unconnected_v_pairs[neg_sel]) 
```

We'll also want to split up our samples so we have a both a training *and* a validation set. If we take the samples we've already generated (instead of generating more), we'll make sure we don't accidentally end up with duplicated pairs in both our training and validation sets (which would kind of defeat the purpose of having a validation set in the first place).

```python
split = int(2500*0.2) # Pick an index to split at

pos_train, pos_valid = pos_sample[:split],pos_sample[split:]
neg_train, neg_valid = neg_sample[:split],neg_sample[split:]

train_list = pos_train + neg_train
valid_list = pos_valid + neg_valid
```
</div>

#### Transformed Lists, DataLoaders
<div class="cblock">

Now to turn the samples into `TfmdLists`, with the correct transform behavior depending on if we're training or validating:

```python
train_tl = TfmdLists(train_list, NYearsTransform(arr_dict,mode='train',n=3))
valid_tl = TfmdLists(valid_list, NYearsTransform(arr_dict,mode='eval',n=3))
```
Not a long step 😁

Now that we have our transformed lists, they function just like a fast.ai `Dataset` object, which means we can actually pass them straight into a `DataLoaders` and get training!

```python
dls = DataLoaders.from_dsets(train_tl, valid_tl)
dls = dls.cuda() # If you're using a GPU (probably a good idea)
```
</div>

## Training
<div class="cblock">

With all of that setup done, we can create our model/learner, give it some metrics, and let it rip! The competition's performance metric is ROC AUC, so it might be handy to use that as a metric while training. Fast.ai has this as a built-in metric already, so we can just initialize it and add it straight in to our learner:

```python
roc_auc = RocAucBinary()
graph_learner = cnn_learner(dls,
                            xse_resnext18, # You could slot in any vision model here
                            metrics=[roc_auc,accuracy],
                            pretrained=False,
                            # n_in=2 for two input "channels; n_out=2 for binary classification
                            n_in=2,n_out=2,
                            loss_func=CrossEntropyLossFlat())
```
And now, we train!

```python
graph_learner.fit_one_cycle(5) # Train for 5 epochs.
```
</div>

## Results; or, not everything is a vision problem.
<div class="cblock">

For brevity, I'll spare you the play-by-play process of validating the model against the solution set provided. The real question is, how did it do? At the end of training, the final ROC AUC score was...

```
ROC_AUC: 0.7305075966006858
```
which is quite a bit lower than the tutorial solution's performance.

But that's okay! Not every approach is going to work for any given problem, and I got to dig a lot deeper into the fast.ai framework in the process of bringing the idea to fruition. Looking back over the setup and execution, I can also see some potential areas of improvement that might bring the model up to a more competitive level.
</div>