// Enable syntax highlighting
const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");

module.exports = function(eleventyConfig) {

  // Don't 11tyignore gitignore'd files by default
  // (may require a .eleventyignore file later).
  eleventyConfig.setUseGitIgnore(false);

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
  eleventyConfig.addFilter(
    "formatDate", 
    (dateObj, locales, options) => {
      const formatter = new Intl.DateTimeFormat(
        locales,
        options
      )
      return formatter.format(dateObj);
    }
  );

  eleventyConfig.addFilter(
    "dateISO", 
    (dateObj) => {
      return dateObj.toISOString();
    }
  );

  eleventyConfig.addFilter(
    "regex_extract", 
    (str, pattern) => {
      const re = new RegExp(pattern);
      return re.exec(str);
    }
  );

  // Create 'post' and 'link' collections based on whatever is in the directories.
  // h/t https://www.mattmcadams.com/posts/2022/working-with-11ty-collections/
  function setLayout(item, layout) { // item: obj, layout: string
    // Some real freakin shenanigans to automatically add layout to each page.
    const layout_obj = {layout: `${layout}.njk`}; // TODO: Make extension configurable.
    return Object.assign(item, {data: Object.assign(item.data, layout_obj)})
  }

  function addEntityCollection(conf, name) { // conf: 11tyConfig, name: str
    const glob = `./src/${name}/*.md` // Hardcoded for now, sorry!

    conf.addCollection(
      name,
      async (collectionAPI) => {
        return collectionAPI.getFilteredByGlob(glob).map(
          item => setLayout(item, name)
        );
      }
    );
  }

  // Not sure if this is the cool way to Javascript...
  const entities = ["post", "link"]
  for (const ent of entities) {
    addEntityCollection(eleventyConfig, ent)
  }

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