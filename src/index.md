---
layout: page.njk
title: Home
---
## About me
<div class="cblock">

I'm a [Human Language Technology](https://linguistics.arizona.edu/master-science-human-language-technology-hlt) student at the University of Arizona. Some other tidbits:
- BA in Comparative Literature
- Interested in language, deep learning, graphs, information retrieval, and wrapping my head around new things.
- I live in Phoenix, AZ with my wife and two [heelers](https://en.wikipedia.org/wiki/Australian_Cattle_Dog) (one of each flavor).
- I'm looking for an work! If LinkedIn is your thing, you can find my profile [here](https://linkedin.com/in/jacksonmostoller).

You can read more about what I've been up to on my [blog](/blog) or in my [notes](/notes).

</div>

<h2><a class="hidden" href="/blog">Blog</a></h2>
<div class="cblock">

<ul>
{%- for blog in collections.blog -%}
  <li><a href="{{ blog.url }}">{{ blog.data.title }}</a></li>
{%- endfor -%}
</ul>
</div>

<h2><a class="hidden" href="/notes">Open notes</a></h2>
<div class="cblock">

These are loose ideas, projects, or blogs-in-progress in various states of presentability. Ideally they might make their way to the [blog](/blog)—but for now, they're yours to read!
<ul>
{%- for note in collections.notes reversed -%}
  <li><a href="{{ note.url }}">{{ note.data.title }}</a>, <small><code>{{ note.date.toLocaleDateString }}</code></small></li>
{%- endfor -%}
</ul>
</div>