---
layout: default
title: RIMEL BOOK
subtitle: Reverse Engineering
---

<div class="span12">
<span>

<div class="span7">
This course aims to introduce Software maintenance by illustrative examples and the state of the art. <br/>
In this course, teams of 3-4 students works on a Software maintenance problem. 
During a 7 week period, the students spent 4 hours by weeks on this course. <br/>
Students write this book, in English or French, explaining their own experiments.<br/>

This work draws heavily on  Delft University of Technology in  <a href="https://delftswa.gitbooks.io/desosa2016/content/">2016</a>  and
<a href="https://github.com/delftswa2018/desosa2018">2018</a>.
<br/>
<br/>
<br/>
</div>



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
