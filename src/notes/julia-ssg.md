---
layout: post.njk
title: Julia Static Site Generator
tags: ["notes"]
date: 2022-01-29
---
Right now this site is built using [Eleventy](https://11ty.dev), but I'm also *really* interested in getting into the [Julia](https://julialang.org) language. So I wonder…could I piece together something like that in Julia as a kind of getting-to-know-you project?

Julia seems to have some benefits that might be helpful here:

1. **Type dispatching** could be a really interesting way to handle the semantic content of HTML. Type-dispatched style sheets? Type-dispatched templates? Could be neat!
2. One of the great things about Eleventy is its **flexibility**—it doesn't want to be a framework you have to learn, it wants to be a tool that can play nicely with whatever you already know/use. That same approach could maybe be facilitated by leveraging Julia's [interoperability](https://techytok.com/lesson-other-languages/) to let you use existing frameworks/templating systems/setups from a variety of languages (e.g. JavaScript).