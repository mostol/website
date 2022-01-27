---
layout: page.njk
title: Blog Home
---
This is the main page for my blog, which is where more polished thoughts go to…live.

<ul>
{%- for blog in collections.blog -%}
  <li><a href="{{ blog.url }}">{{ blog.data.title }}</a></li>
{%- endfor -%}
</ul>