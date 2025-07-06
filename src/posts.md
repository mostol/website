---
layout: splash.njk
title: Posts
---
# {{ title }}
Some *rerum vulgarium fragmenta*. A pile of digital loose leaf notes.

This is the firehose—to help myself actually write stuff out, the bar is fairly low for a post. More polished things might also appear [in a different collection](/topics/).

<ul>
{% assign sorted_posts = collections.post | sort: "data.modified" | reverse %}
{%- for post in sorted_posts -%}
  <li>
    <a href="{{ post.url }}">{% if post.data.title %}{{ post.data.title }}{% elsif post.page.fileSlug %}{{ post.page.fileSlug }}{% endif %}</a>
    <small>
    (⊕ {{ post.data.created | formatDate: "en-US" }}{% if post.data.modified > post.data.created %}
    Δ {{ post.data.created | formatDate: "en-US" }}{% endif %})</small>

  </li>
{%- endfor -%}
</ul>