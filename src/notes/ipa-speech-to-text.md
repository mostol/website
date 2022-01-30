---
layout: post.njk
title: IPA Speech-to-Text Model…?
tags: ["notes"]
date: Last Modified
---
Here's a thought: we've got [speech-to-text](https://huggingface.co) models galore. We can take audio input from dozens (now hundreds?) of languages and feed it into a model, and out comes some text that matches. [What a time to be alive!](https://twitter.com/twominutepapers) But hey, those strings of text that the models spit out—what if those were *IPA* letters, instead of specifically Kazakh or French or Navajo or another particular alphabet's representation? That might be neat! Let's see what we can do...

## Getting data
We should really probably use a pre-trained model, but in theory it seems like if a model is already trained on human language in general, it should be okay with a taking variety of languages' acoustic inputs. The trickier part is getting an annotated dataset that has audio labelled in IPA characters. One potential source: the [Wiktionary](https://wiktionary.org/], which conveniently has recordings *and* IPA representations (though not necessarily *precise transcriptions*) from words in hundreds of languages. 

[Here's a chart of all of the IPA characters](https://www.internationalphoneticassociation.org/IPAcharts/IPA_chart_orig/pdfs/IPA_Kiel_2020_symbollist.pdf), which will probably be helpful.