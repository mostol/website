---
layout: post.njk
title: Eatiquette ingredient task overview
date: 2022-02-14
---

<div class="cblock">
With the massive proliferation of machine learning techniques, there are probably infinite possible approaches to the ingredient-mapping task. After some consideration, I feel that some kind of transformer-based approach might turn out to be among the easiest to implement and most elegant.

Transformer-centric models are doing incredible things right now, and it wouldn't be all that difficult to get a transformer up-and-running that could accomplish this labelling task with a good amount of accuracy. But what would that look like?

My inital instict would perhaps be to look at some kind of classifier, i.e. "tomato sauce" and "tomato paste" both get classified as `tomato`, "baking soda" and "SODIUM BICARBONATE" are classified as `sodium bicarbonate`, and so forth. The issue with this approach is that we're looking at *at least* 3000-4000 "classes"—which would mean we would need at least a few examples of sometimes-very-rare ingredients, because instrisically a class' label is meaningless, so we would need concrete examples in each instance.
</div>

## A Sequence-to-sequence approach
<div class="cblock">
A more promising option that would require less input data and far-fetched samples would be to create a sequence-to-sequence model that is, in effect, a "summarizer"—taking in full/long versions of ingredients and returning their summed-up or essentialized version. Just like article-summarization models, this kind of micro-summarizer should be able to tackle new, unseen inputs without a problem. Here's how this kind of model would handle the three components outlined in the task description:

1. *Identify core ingredients.* The seq-to-seq model would be trained to convert a non-standardized ingredient into a standardized format (e.g. "whole wheat flour" to `wheat`). The core ingredients list would then consist of the set of all outputs from the model for every one of the 300,000+ ingredients in the non-standardized list, which should be able to convert any given ingredient-like string into its "core" ingredient (since that's what it's been trained to do ☺️).
2. *Map ingredients to the list of core ingredients.* Step 2 is completed in essentially the same breath as step one: the model learns to map an ingredient to its "core" version, and the total output of all ingredients forms the canonical "core" list.
3. *How will the model work for a new set of ingredients?* The model should have been trained without seeing *all* ingredient examples (i.e. using a validation set), so if it had been trained correctly it will have properly generalized the notion of "tell me the core ingredient of this nonstandard ingredient listing," and will be able to do so for ingredients that it has not yet seen. This should especially be feasible if the model is based on an existing pre-trained model, which will still be familiar with concepts and vocabulary beyond the scope of the training set and can generalize this preexisting knowledge to tackle never-before-seen ingredients.

Using a pre-trained transformer-based language mode allows us to leverage highly sophisticated linguistic connections between words, and also enables high generalizability which helps the model handle data issues such as spelling errors or irrelevant information (like extended but unnecessary marketing descriptions of ingredients).
</div>

## Some considerations

<div class="cblock">
While this approach seems to check all of the boxes, there are a few considerations to tak into account, as there would be with nearly any approach. One aspect of this kind of transformer-based sequence-to-sequence approach is the need for labelled training data, which means that 1) up to several thousand examples may need to be labelled by hand before the model can function correctly, and 2) because this model runs on labelled data in a supervised fashion, certain design decisions need to be made before hand and become somewhat inflexible once the model is trained. For example, what do we want to count as an "ingredient"? How different are "soy" and "soy sauce"? What about "beef" and "beef broth"? If there are multiple ingredients in one input, should we be able to detect that and reduce things to multiple core ingredients? With this kind of supervised learning approach, we need to make these calls at the beginning—which is fine!—but if we want to change our decisions later we may have to do some re-training or re-evaluating. 

There are upsides to these ramifcations as well, however. For example, the task description notes that "Process identifiers from an ingredient like ‘dried’, ‘roasted’, ‘nonGMO’, ‘gluten-free’ should be treated separately." We could take this to mean that "dried", "roasted", "non-GMO", and "gluten-free" are to be *unconsidered* or *ignored* by the model, because we only want the ingredients. But, aligning with the Eatiquette matra:

> Food taste, preference, and medical avoidance is highly individual. We are not here to push for a certain lifestyle and try to stay away from opinions on what you should and should not eat.

Some of these pieces of information may be useful to keep track of, even if not necessarily mandated by the current task constraints. Users may, for example, want to know about organic ingredients, or their geographic origins, or whether something is whole-grain, or even an ingredient's preparation method. Since the precise way in which data is labelled is important and will require meaningful forethought, we could take this opportunity to label the data and train the model to extract as much meaningful information as possible and open up new possibilities.
</div>