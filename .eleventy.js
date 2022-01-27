module.exports = function(eleventyConfig) {
  // Copy `src/style.css` to `_site/style.css`
  eleventyConfig.addPassthroughCopy("src/style.css");

  // Copy `fonts` to `_site/fonts`
  eleventyConfig.addPassthroughCopy("fonts");

  // Copy `img` to `_site/img`
  eleventyConfig.addPassthroughCopy("img");

  return {
    // When a passthrough file is modified, rebuild the pages:
    passthroughFileCopy: true,
    dir: {
      input: "src",
      includes: "_includes",
      data: "_data",
      output: "_site"
    }
  };
};
