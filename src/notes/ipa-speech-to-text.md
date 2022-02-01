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

[Here's some Julia package handling info.](https://blog.devgenius.io/the-most-underrated-feature-of-the-julia-programming-language-the-package-manager-652065f45a3a)

```julia
using JSON3

function get_words(file::String)
    a::Array{AbstractDict} = []
    for line in eachline(file)
        if occursin(r"\"ipa\"",line) && occursin(r"\"(mp3|ogg)_url\"")
            push!(a,JSON3.read(line).sounds)
        end
    end
    return a
end
```

Other things to improve:
* We probably want to narrow it down to items with *one* pronunciation, or we can't quite tell if we have the right pronunciation transcribed. (If we can find a way to differentiate e.g. between US and UK pronunciations and match each with the correct transcription, that'd be 💯)
* There are (generally) two transcriptions: a [phonemic and a phonetic](https://linguistics.stackexchange.com/questions/301/when-should-one-use-slashes-or-square-brackets-when-transcribing-in-ipa), the phonemic (in square brackets) being the more narrow one in general. It's maybe preferable to grab that one but then again—why not both? It might be handy to have either option available, or to compare and see which is a more successful basis for a model.