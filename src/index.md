---
layout: page.njk
title: Home
---
## About me
<div class="cblock">

I'm a Human Language Technology student at the University of Arizona. Some other details are:
- BA in Comparative Literature (plus a Mathematics minor)
- Interested in language, deep learning, graphs, information retrieval, web design/development, and wrapping my head around new things. 
- I live in Phoenix, AZ with my wife and two heelers (one of each flavor).
- I'm looking for an internship 😉

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
{%- for note in collections.notes -%}
  <li><a href="{{ note.url }}">{{ note.data.title }}</a></li>
{%- endfor -%}
</ul>
</div>