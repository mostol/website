---
layout: splash.njk
title: Links
---
# {{ title }}
Grab-bag of stuff across the internet.

<ul>
{% assign sorted_links = collections.link | sort: "data.created" | reverse %}
{%- for link in sorted_links -%}
  <li>
    <a href="{{ link.url }}">{% if link.data.title %}{{ link.data.title }}{% elsif link.page.fileSlug %}{{ link.page.fileSlug }}{% endif %}</a>
    <dl class="metas">
      <dt>↗</dt>
      <dd><a href="{{ link.data.url }}" target="_blank" rel="external">{{ link.data.url | regex_extract: "(?<=:\/\/)[^\/]+" }}</a></dd>
      <dt>⊕</dt>
      <dd>{{ link.data.created | formatDate: "en-US" }}</dd>{% if link.data.tags %}
      <dt>⌗</dt>
      <dd>
      {{ link.data.tags | join: " " | }}
      </dd>{% endif %}
    </dl>
  </li>
{%- endfor -%}
</ul>