---
layout: post.njk
title: IPA Speech-to-Text Model…?
tags:
    - "post"
date: 2022-01-29
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
        if occursin(r"\"ipa\"",line) && occursin(r"\"(mp3|ogg)_url\"",line)
            push!(a,JSON3.read(line).sounds)
        end
    end
    return a
end
```

```
{"pos": "name", "head_templates": [{"name": "head", "args": {"1": "es", "2": "proper nouns", "head": "", "g": "?", "g2": "", "g3": ""}, "expansion": "Llanquihue ?"}, {"name": "es-proper noun", "args": {}, "expansion": "Llanquihue ?"}], "word": "Llanquihue", "lang": "Spanish", "lang_code": "es", "sounds": [{"ipa": "/\u029dan\u02c8kiwe/"}, {"ipa": "[\u025f\u0361\u029d\u00e3\u014b\u02c8ki.we]"}, {"ipa": "/\u029dan\u02c8kiwe/"}, {"ipa": "[\u025f\u0361\u029d\u00e3\u014b\u02c8ki.we]"}, {"ipa": "/\u028ean\u02c8kiwe/"}, {"ipa": "[\u028e\u00e3\u014b\u02c8ki.we]"}, {"ipa": "/\u0283an\u02c8kiwe/"}, {"ipa": "[\u0283\u00e3\u014b\u02c8ki.we]"}, {"ipa": "/\u0292an\u02c8kiwe/"}, {"ipa": "[\u0292\u00e3\u014b\u02c8ki.we]"}, {"audio": "LL-Q1321 (spa)-Millars-Llanquihue.wav", "text": "Audio (Spain)", "tags": ["Spain"], "ogg_url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/5/51/LL-Q1321_%28spa%29-Millars-Llanquihue.wav/LL-Q1321_%28spa%29-Millars-Llanquihue.wav.ogg", "mp3_url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/5/51/LL-Q1321_%28spa%29-Millars-Llanquihue.wav/LL-Q1321_%28spa%29-Millars-Llanquihue.wav.mp3"}], "senses": [{"raw_glosses": ["A province of Chile"], "glosses": ["A province of Chile"], "id": "Llanquihue-es-name-ICxVvTSj", "categories": [{"name": "Places in Chile", "kind": "place", "parents": ["Places", "List of sets", "Names", "All sets", "Proper nouns", "Terms by semantic function", "All topics", "Fundamental", "Lemmas"], "source": "w", "orig": "es:Places in Chile", "langcode": "es"}, {"name": "Provinces of Chile", "kind": "place", "parents": ["Provinces", "Places", "List of sets", "Political subdivisions", "Names", "All sets", "Polities", "Proper nouns", "Terms by semantic function", "All topics", "Fundamental", "Lemmas"], "source": "w", "orig": "es:Provinces of Chile", "langcode": "es"}]}]}

{"pos": "name", "head_templates": [{"name": "head", "args": {"1": "es", "2": "proper nouns", "head": "", "g": "?", "g2": "", "g3": ""}, "expansion": "Risaralda ?"}, {"name": "es-proper noun", "args": {}, "expansion": "Risaralda ?"}], "word": "Risaralda", "lang": "Spanish", "lang_code": "es", "sounds": [{"ipa": "/risa\u02c8\u027ealda/"}, {"ipa": "[ri.sa\u02c8\u027eal\u032a.d\u032aa]"}, {"audio": "LL-Q1321 (spa)-Millars-Risaralda.wav", "text": "Audio", "ogg_url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/6/62/LL-Q1321_%28spa%29-Millars-Risaralda.wav/LL-Q1321_%28spa%29-Millars-Risaralda.wav.ogg", "mp3_url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/6/62/LL-Q1321_%28spa%29-Millars-Risaralda.wav/LL-Q1321_%28spa%29-Millars-Risaralda.wav.mp3"}], "senses": [{"raw_glosses": ["A department of Colombia"], "glosses": ["A department of Colombia"], "id": "Risaralda-es-name-0Pwvq-oJ", "categories": [{"name": "Departments of Colombia", "kind": "place", "parents": ["Departments", "Places", "List of sets", "Political subdivisions", "Names", "All sets", "Polities", "Proper nouns", "Terms by semantic function", "All topics", "Fundamental", "Lemmas"], "source": "w", "orig": "es:Departments of Colombia", "langcode": "es"}, {"name": "Places in Colombia", "kind": "place", "parents": ["Places", "List of sets", "Names", "All sets", "Proper nouns", "Terms by semantic function", "All topics", "Fundamental", "Lemmas"], "source": "w", "orig": "es:Places in Colombia", "langcode": "es"}]}]}

{"pos": "noun", "head_templates": [{"name": "es-noun", "args": {"1": "mf", "f": "jueza"}, "expansion": "juez m or f (plural jueces, feminine jueza, feminine plural juezas)"}], "forms": [{"form": "jueces", "tags": ["plural"]}, {"form": "jueza", "tags": ["feminine"]}, {"form": "juezas", "tags": ["feminine", "plural"]}], "wikipedia": ["Diccionario cr\u00edtico etimol\u00f3gico castellano e hisp\u00e1nico"], "etymology_text": "From Old Spanish juez, juiz, judez, from Latin i\u016bdex, j\u016bdex or i\u016bdicem, j\u016bdicem; possibly a semi-learned term.", "etymology_templates": [{"name": "inh", "args": {"1": "es", "2": "osp", "3": "juez"}, "expansion": "Old Spanish juez"}, {"name": "m", "args": {"1": "osp", "2": "juiz"}, "expansion": "juiz"}, {"name": "m", "args": {"1": "osp", "2": "judez"}, "expansion": "judez"}, {"name": "der", "args": {"1": "es", "2": "la", "3": "iudex", "4": "i\u016bdex, j\u016bdex"}, "expansion": "Latin i\u016bdex, j\u016bdex"}, {"name": "m", "args": {"1": "la", "2": "iudicem", "3": "i\u016bdicem, j\u016bdicem"}, "expansion": "i\u016bdicem, j\u016bdicem"}, {"name": "w", "args": {"1": "Diccionario cr\u00edtico etimol\u00f3gico castellano e hisp\u00e1nico"}, "expansion": "Diccionario cr\u00edtico etimol\u00f3gico castellano e hisp\u00e1nico"}], "sounds": [{"ipa": "/\u02c8xwe\u03b8/"}, {"ipa": "[\u02c8xwe\u03b8]"}, {"ipa": "/\u02c8xwes/"}, {"ipa": "[\u02c8xwes]"}, {"audio": "LL-Q1321 (spa)-AdrianAbdulBaha-juez.wav", "text": "Audio (Colombia)", "tags": ["Colombia"], "ogg_url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/6/63/LL-Q1321_%28spa%29-AdrianAbdulBaha-juez.wav/LL-Q1321_%28spa%29-AdrianAbdulBaha-juez.wav.ogg", "mp3_url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/6/63/LL-Q1321_%28spa%29-AdrianAbdulBaha-juez.wav/LL-Q1321_%28spa%29-AdrianAbdulBaha-juez.wav.mp3"}], "word": "juez", "lang": "Spanish", "lang_code": "es", "senses": [{"raw_glosses": ["judge"], "tags": ["feminine", "masculine"], "glosses": ["judge"], "id": "juez-es-noun-EOhsZRTU"}, {"raw_glosses": ["umpire; referee; official"], "tags": ["feminine", "masculine"], "glosses": ["umpire; referee; official"], "id": "juez-es-noun-J8f61gtZ", "categories": [{"name": "Law", "kind": "topical", "parents": ["Society", "All topics", "Fundamental"], "source": "w+disamb", "orig": "es:Law", "langcode": "es", "_dis": "8 92"}, {"name": "Occupations", "kind": "topical", "parents": ["List of sets", "People", "All sets", "Human", "Fundamental", "All topics"], "source": "w+disamb", "orig": "es:Occupations", "langcode": "es", "_dis": "3 97"}]}]}
```

```julia
julia> function get_words_regex(file::String)
           a::Array{Dict} = []
           for line in eachline(file)
               ipa_sl = collect(eachmatch(r"(?<=\"ipa\": \")\/.*?\/",line))
               ipa_br = collect(eachmatch(r"(?<=\"ipa\": \")\[.*?\]",line))
               url = collect(eachmatch(r"(?<=\"\w{3}_url\": \").*?(?=\")",line))
               if length(ipa_br) == 1 && length(ipa_sl) == 1 && length(url) > 0
                   word = match(r"(?<=\"word\": \").*?(\?=\")",line)
                   push!(a,Dict("word"=>word,"phonemic"=>ipa_sl,"phonetic"=>ipa_br))
               end
           end
           return a
       end
```

Other things to improve:
* We probably want to narrow it down to items with *one* pronunciation, or we can't quite tell if we have the right pronunciation transcribed. (If we can find a way to differentiate e.g. between US and UK pronunciations and match each with the correct transcription, that'd be 💯)
* There are (generally) two transcriptions: a [phonemic and a phonetic](https://linguistics.stackexchange.com/questions/301/when-should-one-use-slashes-or-square-brackets-when-transcribing-in-ipa), the phonemic (in square brackets) being the more narrow one in general. It's maybe preferable to grab that one but then again—why not both? It might be handy to have either option available, or to compare and see which is a more successful basis for a model.

```
https://upload.wikimedia.org/wikipedia/commons/6/63/LL-Q1321_%28spa%29-AdrianAbdulBaha-juez.wav
https://upload.wikimedia.org/wikipedia/commons/6/63/LL-Q1321_%28spa%29-AdrianAbdulBaha-juez.wav
https://upload.wikimedia.org/wikipedia/commons/transcoded/6/63/LL-Q1321_%28spa%29-AdrianAbdulBaha-juez.wav
```

Okay! So, I *finally* know how to get a URL from a WikiMedia Commons file name (thanks [StackOverflow](https://stackoverflow.com/questions/30781455/how-to-get-image-url-in-wiki-api)). So what I want is this:

```
https://commons.wikimedia.org/wiki/Special:FilePath/LL-Q1321_%28spa%29-AdrianAbdulBaha-juez.wav
```