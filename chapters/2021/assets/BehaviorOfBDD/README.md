# Rimel Team F - Subject 2

This repository is an runner, analyser and generator of results for SI5-Rimel Project

## Context

We want to answer the question "What's is the link between Fonctionnal Tests and Unit Tests ?"
To answer this question, we have 2 subquestions for which this project tries to find an answer :

-   Do units tests and fonctionnal tests test the same places in the code or are they complementary?
-   Is there a link between the number of unit tests calling a method and the number of functionnal tests calling this method?

## Output

This project generates for each project these files:

-   `matrix-tu.json` and `matrix-tf.json` : The coverage of each line of code for each unit test and functionnal test
-   `method-list.json` : The list of methods of the business logic layer (in the main folder)
-   `matrix-methods-tu.json` and `matrix-methods-tf.json` : The coverage of each method for each tests
-   `matrix-methods-sum-tu.json` and `matrix-methods-sum-tf.json` : The number of unit and functionnal tests covering each method
-   `matrix-methods-sum-merged.json` : The number of unit and functionnal test covering each method in this one file

## Requirement

To use our script you must add Jacoco inside each project to be analyzed, to do that, copy and paste this plugin inside the main pom.xml :

```xml
<plugin>
    <groupId>org.jacoco</groupId>
    <artifactId>jacoco-maven-plugin</artifactId>
    <version>0.8.2</version>
    <executions>
        <execution>
            <goals>
                <goal>prepare-agent</goal>
            </goals>
        </execution>
        <execution>
            <id>report</id>
            <phase>test</phase>
            <goals>
                <goal>report</goal>
            </goals>
        </execution>
    </executions>
</plugin>
```

You must have `node 14` and `npm or yarn`

## Install

For each following project

```
CoverageParser
DirectoryParser
ExtensionModifier
MethodsParser
```

you must install all dependencies with

```bash
npm install
```

or

```bash
yarn
```

And now install dependencies in the `MainProject`

```bash
npm install
```

or

```bash
yarn
```

## How to use

Go to MainParser and type and replace path with the path of the project that you want to analyse and output with the directory where you want to generate the output files

```bash
node app.js --path={path} --all --output={output}
```

## Generate graphic

If you want generate graphic you can go to `GraphicGenerator` and install dependencies with

```bash
npm install
```

or

```bash
yarn
```

and run and replace input with the output folder written above and output with the name of graph

```bash
node index.js {input} -o {output} --dir
```

To generate Venn plot and scatter plot from the results of previous scripts, execute the “scripts.py” file from the “DataVizScripts” folder (with the command “python scripts.py” or from IDLE).
The plots will be generated graphs will be in the "output" folder next to the outputs of other scripts.  
