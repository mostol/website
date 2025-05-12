---
layout: splash.njk
title: Topics
---
# {{ title }}
A list of all collections/topics. All posts are also available [here](/posts/).

<ul>
{%- for topic in collections.topic -%}
  <li>
  <a href="{{ topic.url }}">{{ topic.data.title }}</a>.
  <small> {{ topic.data.description }} </small>
  </li>
{%- endfor -%}
</ul>