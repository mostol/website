### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 9600e48a-62b3-11ec-24f8-bf87a18a6afe
content = Markdown.html(Markdown.parse("#Hello"))

# ╔═╡ 7742452f-b234-4541-b206-28f01a3eeb52
Markdown.html(Markdown.parse_file("index.md"))

# ╔═╡ 51f9e12f-c15e-4e59-8279-2f2a713b82ed
contents = read("index.md",String)

# ╔═╡ e974326c-cba5-465f-aad4-b6702498e151


# ╔═╡ f748ee7e-8869-44a7-b908-1796d91dc768
collect(eachmatch(r"# \N*"s,contents))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[deps]
"""

# ╔═╡ Cell order:
# ╠═9600e48a-62b3-11ec-24f8-bf87a18a6afe
# ╠═7742452f-b234-4541-b206-28f01a3eeb52
# ╠═51f9e12f-c15e-4e59-8279-2f2a713b82ed
# ╠═e974326c-cba5-465f-aad4-b6702498e151
# ╠═f748ee7e-8869-44a7-b908-1796d91dc768
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
