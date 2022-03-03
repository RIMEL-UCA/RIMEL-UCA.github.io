# si5-rimel-github-parser

## Run the project
 - Wrap download / categorize / analyze
`npm run run -- --help`

 - Download
`npm run download -- --help`

- Scan
`npm run scan -- --help`

- Categorize
`npm run categorize`

- Analyze DevDependencies
`npm run analyze:1`

- Analyze EsLint
`npm run analyze:2` 


### Limitations
- The repositories are taken by order of updated date, not randomly.
- It is difficult to retrieve native webpack projects (non-framework) via the github api, we get false positives.