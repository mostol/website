---
layout: splash.njk
title: Blogs
tags:
- "topic"
description: Longer-form posts.
---
# {{ title }}
A place for more polished thoughts.

<ul>
{%- for blog in collections.blog reversed -%}
  <li>
    <a href="{{ blog.url }}">{{ blog.data.title }}</a>
    <small>({{ blog.data.date | formatDate }})</small>
  </li>
{%- endfor -%}
</ul>