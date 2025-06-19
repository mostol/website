---
layout: splash.njk
title: Links
---
# {{ title }}
Grab-bag of stuff across the internet!

<ul>
{%- for link in collections.link -%}
  <li>
    <a href="{{ link.url }}">
    {% if link.title %}{{ link.title }}{% elsif link.page.fileSlug %}{{ link.page.fileSlug }}{% endif %}
    </a>
    <small>({{ post.data.date | formatDate }})</small>
  </li>
{%- endfor -%}
</ul>