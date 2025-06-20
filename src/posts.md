---
layout: splash.njk
title: Posts
---
# {{ title }}
Some *rerum vulgarium fragmenta*. A pile of digital loose leaf notes.

This is the firehose—to help myself actually write stuff out, the bar is fairly low for a post. More polished things might also appear [in a different collection](/topics/).

<ul>
{%- for post in collections.post reversed -%}
  <li>
    <a href="{{ post.url }}">
    {% if post.data.title %}{{ post.data.title }}{% elsif post.page.fileSlug %}{{ post.page.fileSlug }}{% endif %}
    </a>
    <small>({{ post.data.date | formatDate }})</small>
  </li>
{%- endfor -%}
</ul>