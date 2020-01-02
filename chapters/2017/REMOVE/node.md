# **Analysis of a large-scale multi-language software systems applied to Node Js**

## **Cases Study**

The node project was created during 2009 by Ryan Dahl, it is one of the most followed projects currently on github with more than 1200 contributors. More than 16,000 commits have been carried out on the Node project since its inception. It has been developed in several languages in C / C ++ and Javascript. The main question is, are there people responsible for several modules written in both languages? If yes, what is their percentage? How could they qualify? To answer these same questions in other multi-language projects, we propose to follow our approach which will be described in this document.

## **Tools and Method**

In this section, we will describe step by step the tools and methods we will use to answer the question we asked.

The first phase of our analysis is to generate the input data.To analyze our VCS data we need to define a temporal period of interest. We will use the git log command to generateVCS data.

Once we have the data, we will remove the data that is not necessary for our analysis. To do this we will use our own parser which takes an input log file as well as a list of extensions and outputs a filtered log.

Then we will analyze the file obtained previously using codeMaat. We will mainly use the summary option to give an overview of the organizational metrics of the Node project.

The final phase of our analysis will be to visualize and interpret the results we have obtained. We will use a graphical tool for visualization \(eg Excel\). To interpret our results, we start from the assumptions made in the previous section.

# **Results**

Before answering the main question, we will give some organizational metrics in the project. To do this, we will look at the percentage of commits in each language but also the percentage of contributors per language

* ### Contributors

Below is the percentage of contributors per language

![](https://lh4.googleusercontent.com/MY0Xit7Z93aJTQamgyeNm2JK8UmYOo7ewZKfQ-srhS2SpQaxgxOIdhW8c3n2EW5WJ3OSyYdvEFG2c5ebri_vs9RmER8SoYz91ESP1kEIAOn8K7bxXMT7stRmrfMJL8B9uLsJXILC)

We see in the figure above that there are 1252 contributors. 62% are mainly Javascript developers, 22% are C / C ++ developers and 16% for other contributors \(document, installation script, ...\). We deduce that there are more Javascript contributors than C / C ++ contributors.

* ### **Commits**

**Below is the percentage of commit by language**

![](https://lh6.googleusercontent.com/PBoIbia6t9D6j7YxVGllPWcEmy96WImqu74HE91dFF_2yMZSmfUaSLImNp_fqAjizBYOEnScUNQxHzJs63LYCKgN8zV6sfxRvXbeculwfPJF2HbzxdZzL2tD07he25E3dkHjGXNy)

The figure above shows the percentage of commits in different languages. We find that 50% of the commits concern javascript, 32% of commits concern the C / C ++ and 18% for the others \(Document, Script, ...\).

We notice through the two previous analyses, that there is a consistency between the two metrics. Technically that the percentage of contributor by language follows the number of commits.

Now that we have the organizational metrics in the project in a global way, we will refine our analysis by looking only at the languages ​​that interest us. That is to say we will use our parser to remove unnecessary data \(for example, md file, script, ...\).

![](https://lh4.googleusercontent.com/Mdsvjqw7Jz8pGLbhvBUBvLxhZDs0vlw_qzD5ctl8jGBWKps55DO1_EMslzp9OEEP22uY3eZPK_szPSGV9VGLsO4vw9DVCknT2Cg3yHDXipmz5Y4WcCvfsHeBT0dVu9jgFVaIbuRM)

The figure above gives an estimate of the percentage of the contributions in each language. It was obtained by applying a filter on the commits that are at least one of the language C/C++/JS. We note that the percentage of the contributions in Javascript and C/C++ corresponds respectively to 68% and 32%.

**Are there developper responsible for several modules developed in the different languages ?**

We still more refine our analysis by looking at contributions concerning two languages at the same time. the goal is to find developers who contribute in both languages at the same time, the distribution of their effort and qualify them.

**Percentage of contributors in both languages:**

The figure above shows the percentage of contributors in both languages at the same time.

![](https://lh3.googleusercontent.com/L2s2WU-ZrAYG35_adN4Q_8qj7dJpY8IkPBM1m823aHx94Ann4gwnr5ksRS4VhG8FuHUYu5eabO6LgLRnzyYvH94woNswhaQOzPh0YBMGls06ihN79jKEqX0C8pqPCBh436hQGosk)

The figure below that 11% of developers work in both.

**Percentage of commits in both languages:**

The figure below shows the percentage of commits in both languages at the same time.

![](https://lh3.googleusercontent.com/TyvEA7ctRiPgE6kzM8ANDLAnIvPM7ku3sC3eIIHZdgLP4pX_32FqbL1uztaI2h6Zpdv_0RZhtUTJUTth3VkyhYhzkQg4uo7Cb64IW-W4dblg_MiIplhPzcewtU35r1vt05ps3f5E)

Contrary to the result obtained previously, only 7% of commits apply to both languages.

Now that we know the percentage of contributors in both languages, we will qualify them based on the assumption we made.

Property = percentage of added lines

Minor contributor = property &lt;2%

Major contributor = property &gt; 2%

![](https://lh4.googleusercontent.com/TExM0xmkkUElnYf3dRr9N0_i_rfakp0VmJ2cCWDeU3OKK2wOzZNbFvYJQfbQGpmc6KWgd3Ub4qtDKQcWaoDSFC5BImyw5gCx3VKPuN-54f-JHiWnJhESwKzjkBPRBpBDMuJ-7x_w)

**The figure below shows the distribution of contributor efforts in both languages.**

![](https://lh5.googleusercontent.com/Iyj5qAQongZS6cDIryTctdGlD35lGuWg4Q5-BOMyljC170CV8a3uXH9Q321IE9CEABV8xbEl7EOdjyCnEzj8yYgqqdL3JphvEEPhhASmXdp6ialhpSBFGz5nclpogH6lOyp-BMPY)

In the figure above we observe that there are 6 major contributors.

**Percentage of commits for both languages:**

![](https://lh3.googleusercontent.com/TyvEA7ctRiPgE6kzM8ANDLAnIvPM7ku3sC3eIIHZdgLP4pX_32FqbL1uztaI2h6Zpdv_0RZhtUTJUTth3VkyhYhzkQg4uo7Cb64IW-W4dblg_MiIplhPzcewtU35r1vt05ps3f5E)

The diagrams below show that 7% of the commits concern the two languages.

Now let's look at the major contributors based this time on another property.

Property = percentage of commits

Minor contributor = property &lt; 2%

Major contributor = property &gt; 2%

![](https://lh6.googleusercontent.com/PKof_F4ojoUMgdkWVfzyiubzu8DSOYo04jTl05-AQadKeqOAPj-HMej0dPeIRIs-gRknCIKpwRT5I1v6T6AjsaBRJi_QaXhorpOTSvn-l5otX1lcRnsTmw2SNBKWhpzH1cKNKSSu)

![](https://lh5.googleusercontent.com/snxfn8nEo3122I94lGLXerVqYj-Rqf4wkbWCDtzm4l_qa8AWuPMJNWrqynaj_2fy72QLFf7yiimEbR1m_8DZ6o3vjerMCGJviw-S53M6sp1HLRGaDrmsJ2dF2iQTtdWsudzjt9OP)

In the figure above we observe that there are 7 major contributors.

The question that now arises, is there a correlation between major contributors based on the two properties?

![](https://lh3.googleusercontent.com/8ZSptnM2oMKWEYqao6lwXwfU76MBEopTW2N60Of_hILnoacP32bZH5vBkOwfviondgVG5n4vEGl2lQnDHT8I4sOTBElFBwEnVP0Ql4B9T3WDZM3JauO1vnT8ITHxbORHZdX01dPZ)![](https://lh6.googleusercontent.com/roIKjZ0WybFKj9oQj-XWCG7nEfCPXPO7g8Eygso75apCXvv3qHLllqGY_mQCS7dkpo-0vQbFcjtEkNyuRybaQZzbGdiF1yWVskOAVazN7r9JzAgGE1motjwBsTzV0EjAesnqfbdC)

Even though the number of major contributors in the two is almost identical, we still notice that the person who made the most commit in relation to another person does not necessarily contribute the most. For example Michaël Zasso has committed more Fedor Indutry line, he made fewer commit than Fedor since he does not even appear in the first figure while logic would want it to have contributed more than Fedor.

## Conclusion

This analysis allowed us to understand a large scale software system by looking at the change.

Also note that we have encountered difficulties we overcame. Since the analysis process is a succession of steps to achieve the desired results, we have been confronted with the problem of unavailability of a tool to directly analyze our project. That is why we have been led to develop a script that is based on regular expressions to remove unnecessary data.

While our analysis provided an overview of the organizational metrics of a multi-language project, identifying major contributors, we recognize that it is far from exhaustive. It may still be the basis in the analysis of multi-language projects, this is made possible through our script is highly configurable.

Eventually, we would like to deepen our analysis of the ownership pattern in order to generalize to other multi-language projects.

## Tools

```
         [Codemaat](https://github.com/adamtornhill/code-maat)         

         [Parseur](https://drive.google.com/drive/folders/0B0A0gcBA0qpha0JZMEhtZGFaNGM?usp=sharing)

         Excel 
```

Authors:

Balde Thierno

Diallo Mahmoud

**            
**

