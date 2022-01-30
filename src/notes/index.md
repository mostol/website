---
layout: page.njk
title: Open notes collection
---
# {{ title }}
; or, *rerum vulgarium fragmenta*. A pile of digital loose leaf notes. 📓

<ul>
{%- for note in collections.notes -%}
  <li><a href="{{ note.url }}">{{ note.data.title }}</a></li>
{%- endfor -%}
</ul>