---
layout: post.njk
title: Open Notes Home
---
This is the home for my "Open Notes". These are unpolished notes or ideas or blogs-in-progress that will ideally make their way into something more fleshed out—but for now, they're yours to read.

<ul>
{%- for note in collections.notes -%}
  <li><a href="{{ note.url }}">{{ note.data.title }}</a></li>
{%- endfor -%}
</ul>