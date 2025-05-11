// Enable syntax highlighting
const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");

module.exports = function(eleventyConfig) {

  // Add syntax highlighting
  eleventyConfig.addPlugin(syntaxHighlight);

  // Copy `src/style.css` to `_site/style.css`
  eleventyConfig.addPassthroughCopy("src/style.css");

  // Copy `fonts` to `_site/fonts`
  eleventyConfig.addPassthroughCopy("fonts");

  // Copy `img` to `_site/img`
  eleventyConfig.addPassthroughCopy("img");

  // Custom date filter; see https://karlynelson.com/posts/how-to-handle-dates-in-11ty/
  // I'll note that I don't like having to specially-define this here,
  // and also that the parameters have to be hard-coded...
  eleventyConfig.addFilter("postDate", (dateObj) => {
    const formatter = new Intl.DateTimeFormat(
      "en-US",
      {dateStyle: "short"}
    )
    return formatter.format(dateObj);
  });

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