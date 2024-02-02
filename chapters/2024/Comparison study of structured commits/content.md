---
layout: default
title : Comparison study of structured commits
date:   2024-02
---

**_02 février 2024_**

## Authors

We are four students in M2 or in last year of Polytech’ Nice-Sophia specialized in Software Architecture :

* Hadil AYARI
* Nicolas GUIBLIN
* Chahan MOVSESSIAN
* Floriane PARIS

## I. Research context 

The main question we chose to address is : How and to what extent are structured commits used in open-source projects ?
At first, we wanted to understand the impact of commit standards on the management of software projects (clarity, traceability, maintenance). but we soon realized it is a really vast subject, and most of our interrogations couldn't be answered with the tools we had. How can we attest how useful a structured commit was for a project ? Or how much time was saved thanks to a clear commit message ?
So we changed some questions, and these are the under questions we kept :
    * How common are conventional commits and gitmojis used in open-source projects ?
    * Do employees structure themselves naturally ? Or are there commit conventions on each git-hub repos ?
    * Do regular contributors make more structured commits ?
    * Is there a correlation between the number of contributors and commit conventions ?


## II. General question

None of us knew about conventional commit before starting this project, so we were really curious about it. We soon realized most big open source projects used conventional commits and it is part of a standard, but why ? 
It is indeed a really easy and clear way to communicate changes and fixes. With just a short message, you can say what was the code alteration and why it was made for ; so the message is qualitative and useful for the next developers working on the code. 
Although, even if we never used conventional commits before, we definitely already used structured commits working in companies. So is the strict conventional commit really necessary ?

## III. Information gathering
We gathered information from several sources. We started with the documentation on commit convention that we didn’t know about to be able to recognize it in projects : conventional commits and gitmojis.
We also read the suggested article,  “What makes a good commit message?”. This article defines a good commit message by a message explaining “what” changes were made and “why”. The conventional commit definitely helps to express these two elements in a commit message, so it is a good start for the question : "why is the conventional commit so popular ?".
We collected data from open source Github repositories ; we analyzed them by using "pydriller", a python's module, to create an algorithm able to determine the percentage of structured commits in a project. We used Jupyter notebooks to share codes and graph the commit patterns among different projects. We also studied some smaller projects by hand to test and ensure the accuracy of the tool.
To choose the best data to analyze, we picked big open-source projects that are relatively well-known (filtered by number of stars, watchers, ect…), projects that were able to be easily analyzed and well known companies that have multiple open-source projects.
We decided not to study some projects like Linux or Rust, because it was too consuming for our tool which is meant to be lightweight. Huge Commit number, won’t be able to test and debug our tools for them easily and it would cause some testing issues.
For the initial phase, we’ve not studied projects with unclear structures like React, although we’ve decided to study them for a later question.
 
## IV. Hypothesis & Experiences

* Hypothesis : Influence of structured commits on the maintenance and understanding of projects.
    * Hypothesis 1 : Projects would use the already community-driven norm of conventional commits
    * Hypothesis 2 : Top project contributors usually stick to the commits
    * Hypothesis 3 :  The bigger a project is, the more there is the need for commit conventions within that project.
    * Hypothesis 4 : Companies would use the same commit structure among their top projects to facilitate integration between group members
* Experiments: Quantitative and qualitative analysis of commit practices in various projects.
    * Finding the large projects was relatively easy, finding projects with specific commit structures was tricky. Especially for Gitmojis, we struggled to find any project that used only gitmojis.
    * Struggled with meta projects, as mentioned previously, Meta’s conventions are not public and unclear. Which led us to avoid them at first, but then thought to revisit them. For this, we’ve had to study their commits by hand and get the structure from our own interpretations (for this, it is sure that the accuracy is much lower) 
    * Initially for hypothesis 4, we planned on testing all the projects of a single company with a single commit structure and see if they use the same conventions for all projects. However, it didn’t make sense to do this for apache, as they have explicitly stated their structure for each project. Instead we decided to test how much they use those conventions.


## V. Result Analysis and Conclusion

* Results: Presentation and interpretation of the data collected.
Graphs
    * Companies would rather use their own commit structure rather than conventional commits
    * Even when adapted, Companies usually don’t stick to using community driven commits like conventional-commits or gitmoji and usually end up swapping them.
    * The top contributors are usually more diligent with their commit structure, making sure to adhere to the conventions.
    * Companies with multiple projects (Meta & Apache) don’t use the same conventions for all their projects. It seems like each team personalizes their commits to each team’s requirements
    * Comparing Apache with Meta : Apache is comparatively much stricter and much more explicit about their conventions. Apache has no “ultimate” convention, it defines a convention for each project and they’re strict for each convention per project. This doesn’t seem to be the case with Meta. Much more unconventional or “unprofessional” commits found in Meta’s commits [insert figures for meta unconventional commits]

    * Gitmoji never used (except gitmoji repository or on a short period then canceled)
    * Conventional commit often used BUT derived (adding keywords…)
     * Commits often structured uniquely to the directory
     * Use of automatic commits depending on the project (merge, pull requests, squash, etc.) which makes structuring of commits less necessary because they keep the discussion in the PR of the repo leaders
* Limits : 
    * Recognition of the limitations of the study (sampling, possible bias)
    * Our tool was not perfect, in fact, our tool requires a lot of tweaking depending on the project, since every project had its own conventions.
limitations relating to automated commits vs human made commits. 
    * Uncertainties when it comes to non-public commit information (Meta’s documentation doesn’t specify commits structure and claims that all commits are squashed [meaning automated commits], however studying their commits, we can notice a few obviously non-robotic commits)
* Conclusion: Summary of findings and implications for development practices.
    * Results are inconclusive and are always anecdotal depending on the projects chosen
    * This topic has a lot of nuance (project structure, team organization, developer priorities, outside team communication etc..) and these things we just cannot interpret through just the commit set.


## VI. Tools \(facultatif\)

Talk about how it works in detail : 
Our tool detects conventions by using regex for each convention message. This is simply because most commit conventions require having specific terms at the beginning of a commit.
Our tool also uses Natural Language Processing to detect some other convention patterns. In fact, many projects have explicitly stated that their convention is simply “have a verb at the beginning of the commit”, some have specified the verb to be in Present form, some have specified in past tense, but most have just specified that it just needs to be a verb. 

The way most conventions are tweaked is by modifying the subsystems within the regex as most projects use a variation of that same format.

While we have done our studies on multiple projects, we have kept only the most pertinent results in the notebook provided.


## VI. References

<sub>[Debret 2020] Debret, J. (2020) La démarche scientifique : tout ce que vous devez savoir ! Available at: https://www.scribbr.fr/article-scientifique/demarche-scientifique/ (Accessed: 18 November 2022).
</sub>

* Articles Relevant studies on communication and quality of commit messages. :
    * Tian, Y., Zhang, Y., Stol, K. J., Jiang, L., & Liu, H. (2022, May). What makes a good commit message?. In Proceedings of the 44th International Conference on Software Engineering (pp. 2389-2401). (https://arxiv.org/pdf/2202.02974.pdf)
* Information about structured commits
    * Conventional commits : https:/www.conventionalcommits.org/en/v1.0.0/
    * Gitmoji : https://gitmoji.dev/
* List of open source github projects chosen for statistics (among those with the most commits/contributors)
    * Angular (https://github.com/angular/angular.js) 
    * Nodejs (https://github.com/nodejs/node)
    * fastapi (https://github.com/tiangolo/fastapi)
    * gitmoji (https://github.com/carloscuesta/gitmoji) 
    * React (https://github.com/facebook/react) 
    * flutter (https://github.com/flutter/flutter)
    * Linux (https://github.com/torvalds/linux) 
    * Rust (https://github.com/rust-lang/rust) 
    * Typescript (https://github.com/microsoft/TypeScript)
    * Django (https://github.com/django/django)
    * Nextjs (https://github.com/vercel/next.js) 
    * Cpython (https://github.com/python/cpython)
    * Prometheus (https://github.com/prometheus/prometheus) 
    * Pytorch (https://github.com/pytorch/pytorch) 
    * Git (https://github.com/git/git)

* List of small projects tested by hand to train the tool:
    * Request-promised (https://github.com/request/request-promise) -> 300 commits
    * Deployd (https://github.com/deployd/deployd) -> 1300 commits

![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](assets/images/logo_uca.png){:height="12px"}
