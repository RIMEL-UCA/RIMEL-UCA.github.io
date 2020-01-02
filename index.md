---
layout: default
title: RIMEL BOOK
subtitle: Reverse Engineering
---

<div class="span12">

<span>
<div class="span7">
{% for post in site.posts limit:4 %}
{% include postsummary.html %}
{% endfor %}
</div>
</span>


<div class="span4">

<b>RIMEL</b></br>

<span>
  {% include sidebar_footer.html %}
</span>
</div>

</div>
