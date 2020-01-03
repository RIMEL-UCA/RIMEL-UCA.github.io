---
layout: default
title: RIMEL BOOK
subtitle: Reverse Engineering
---

<div class="span12">
<span>

<div class="span7">
This course aims to introduce Software maintenance by illustrative examples and the state of the art. In this course, teams of 3-4 students works on a Software maintenance problem. During a 7 week period, the students spent 4 hours by weeks on this course. Students will write this book, in English or French, explaining their own experiments.

This work draws heavily on : <a href="https://www.gitbook.com/book/delftswa/desosa2016/details">2016</a>  and 
<a href="https://legacy.gitbook.com/book/delftswa/desosa2018/details">2018</a> 
</div>
</span>


<div class="span7">
{% for post in site.posts limit:4 %}
{% include postsummary.html %}
{% endfor %}
</div>





<span>
  {% include sidebar_footer.html %}
</span>

</span>
</div>
