# my_random_research

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;margin-bottom: 20px;}
.tg td{font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-size:14px;font-weight:bold;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
blockquote { background-color: #cecece !important; padding: 14px 10px 5px 20px !important; border-radius: 5px !important;}
img { border-radius: 5px!important; max-width: 90%; margin: 15px 5%;}
</style>

## Corpus

The corpus contains 339 actions from 34 projects: public-apis/public-apis, vercel/next.js, twbs/bootstrap,
ytdl-org/youtube-dl, vuejs/vue, vuejs/docs, vuejs/router, microsoft/FluidFramework, microsoft/TypeScript,
microsoft/winget-cli, microsoft/fluentui-react-native, microsoft/azuredatastudio, microsoft/vscode,
collet/cucumber-demo, mathiascouste/qgl-template, vitest-dev/vitest, i18next/next-i18next,
jwasham/coding-interview-university, EbookFoundation/free-programming-books, flutter/flutter, mobileandyou/api,
facebook/react, freeCodeCamp/freeCodeCamp, d3/d3, mui/material-ui, trekhleb/javascript-algorithms, mantinedev/mantine,
mattermost/mattermost-server, pynecone-io/pynecone, TheAlgorithms/Python, stefanzweifel/git-auto-commit-action,
axios/axios, raspberrypi/linux, kamranahmedse/developer-roadmap

## Repartition of actions types dataset-wide

- 1.18% GITHUB [ 4/339 ]
- 0.0% INTERNAL [ 0/339 ]
- 0.88% PUBLIC [ 3/339 ]
- 0.0% TRUSTED [ 0/339 ]
- 0.0% FORKED [ 0/339 ]

## Projects

### public-apis/public-apis

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 1.77%      | 6                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name             | Up to date |
|-------------|-------------------------|------------|
| GITHUB      | actions/checkout@v2     | False      |
| GITHUB      | actions/setup-python@v2 | False      |
| GITHUB      | actions/checkout@v2     | False      |
| GITHUB      | actions/setup-python@v2 | False      |
| GITHUB      | actions/checkout@v2     | False      |
| GITHUB      | actions/setup-python@v2 | False      |

#### Precedence

![Precedence test_of_push_and_pull.yml](public-apis/public-apis/precedence/test_of_push_and_pull.png)
![Precedence test_of_validate_package.yml](public-apis/public-apis/precedence/test_of_validate_package.png)
![Precedence validate_links.yml](public-apis/public-apis/precedence/validate_links.png)

#### Dependencies

![Dependencies test_of_push_and_pull.yml](public-apis/public-apis/dependencies/test_of_push_and_pull.png)
![Dependencies test_of_validate_package.yml](public-apis/public-apis/dependencies/test_of_validate_package.png)
![Dependencies validate_links.yml](public-apis/public-apis/dependencies/validate_links.png)

### vercel/next.js

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 4.42%      | 15                |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 2.65%      | 9                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                         | Up to date |
|-------------|-------------------------------------|------------|
| GITHUB      | actions/checkout@v3                 | True       |
| GITHUB      | actions/setup-node@v3               | True       |
| GITHUB      | actions/cache@v3                    | True       |
| GITHUB      | actions/download-artifact@v3        | True       |
| GITHUB      | actions/upload-artifact@v3          | True       |
| PUBLIC      | actions-rs/toolchain@v1             | True       |
| PUBLIC      | EndBug/add-and-commit@v7            | False      |
| PUBLIC      | addnab/docker-run-action@v3         | True       |
| PUBLIC      | vmactions/freebsd-vm@v0             | True       |
| PUBLIC      | datadog/agent-github-action@v1      | True       |
| PUBLIC      | styfle/cancel-workflow-action@0.9.1 | False      |
| PUBLIC      | github/issue-labeler@v2.6           | True       |
| PUBLIC      | dessant/lock-threads@v3             | False      |
| GITHUB      | actions/checkout@v3                 | True       |
| GITHUB      | actions/checkout@v3                 | True       |
| GITHUB      | actions/stale@v4                    | False      |
| GITHUB      | actions/checkout@v3                 | True       |
| GITHUB      | actions/setup-node@v3               | True       |
| GITHUB      | actions/github-script@v6            | True       |
| GITHUB      | actions/checkout@v3                 | True       |
| GITHUB      | actions/cache@v3                    | True       |
| GITHUB      | actions/upload-artifact@v3          | True       |
| GITHUB      | actions/download-artifact@v3        | True       |
| PUBLIC      | addnab/docker-run-action@v3         | True       |

#### Precedence

![Precedence build_test_deploy.yml](vercel/next.js/precedence/build_test_deploy.png)
![Precedence cancel.yml](vercel/next.js/precedence/cancel.png)
![Precedence issue_labeler.yml](vercel/next.js/precedence/issue_labeler.png)
![Precedence issue_lock.yml](vercel/next.js/precedence/issue_lock.png)
![Precedence issue_on_comment.yml](vercel/next.js/precedence/issue_on_comment.png)
![Precedence issue_on_label.yml](vercel/next.js/precedence/issue_on_label.png)
![Precedence issue_stale.yml](vercel/next.js/precedence/issue_stale.png)
![Precedence issue_validator.yml](vercel/next.js/precedence/issue_validator.png)
![Precedence notify_release.yml](vercel/next.js/precedence/notify_release.png)
![Precedence pull_request_stats.yml](vercel/next.js/precedence/pull_request_stats.png)

#### Dependencies

![Dependencies build_test_deploy.yml](vercel/next.js/dependencies/build_test_deploy.png)
![Dependencies cancel.yml](vercel/next.js/dependencies/cancel.png)
![Dependencies issue_labeler.yml](vercel/next.js/dependencies/issue_labeler.png)
![Dependencies issue_lock.yml](vercel/next.js/dependencies/issue_lock.png)
![Dependencies issue_on_comment.yml](vercel/next.js/dependencies/issue_on_comment.png)
![Dependencies issue_on_label.yml](vercel/next.js/dependencies/issue_on_label.png)
![Dependencies issue_stale.yml](vercel/next.js/dependencies/issue_stale.png)
![Dependencies issue_validator.yml](vercel/next.js/dependencies/issue_validator.png)
![Dependencies notify_release.yml](vercel/next.js/dependencies/notify_release.png)
![Dependencies pull_request_stats.yml](vercel/next.js/dependencies/pull_request_stats.png)

### twbs/bootstrap

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 5.01%      | 17                |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 2.95%      | 10                |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                         | Up to date |
|-------------|-------------------------------------|------------|
| GITHUB      | actions/checkout@v3                 | True       |
| GITHUB      | actions/setup-node@v3               | True       |
| GITHUB      | actions/checkout@v3                 | True       |
| GITHUB      | actions/setup-node@v3               | True       |
| GITHUB      | actions/checkout@v3                 | True       |
| PUBLIC      | calibreapp/image-actions@1.1.0      | True       |
| GITHUB      | actions/checkout@v3                 | True       |
| PUBLIC      | github/codeql-action/init@v2        | False      |
| PUBLIC      | github/codeql-action/autobuild@v2   | False      |
| PUBLIC      | github/codeql-action/analyze@v2     | False      |
| GITHUB      | actions/checkout@v3                 | True       |
| PUBLIC      | streetsidesoftware/cspell-action@v2 | True       |
| GITHUB      | actions/checkout@v3                 | True       |
| GITHUB      | actions/setup-node@v3               | True       |
| GITHUB      | actions/checkout@v3                 | True       |
| GITHUB      | actions/setup-node@v3               | True       |
| PUBLIC      | JustinBeckwith/linkinator-action@v1 | True       |
| PUBLIC      | actions-cool/issues-helper@v3       | True       |
| PUBLIC      | actions-cool/issues-helper@v3       | True       |
| GITHUB      | actions/checkout@v3                 | True       |
| GITHUB      | actions/setup-node@v3               | True       |
| PUBLIC      | coverallsapp/github-action@1.1.3    | True       |
| GITHUB      | actions/checkout@v3                 | True       |
| GITHUB      | actions/setup-node@v3               | True       |
| GITHUB      | actions/checkout@v3                 | True       |
| GITHUB      | actions/setup-node@v3               | True       |
| PUBLIC      | release-drafter/release-drafter@v5  | True       |

#### Precedence

![Precedence browserstack.yml](twbs/bootstrap/precedence/browserstack.png)
![Precedence bundlewatch.yml](twbs/bootstrap/precedence/bundlewatch.png)
![Precedence calibreapp-image-actions.yml](twbs/bootstrap/precedence/calibreapp-image-actions.png)
![Precedence codeql.yml](twbs/bootstrap/precedence/codeql.png)
![Precedence cspell.yml](twbs/bootstrap/precedence/cspell.png)
![Precedence css.yml](twbs/bootstrap/precedence/css.png)
![Precedence docs.yml](twbs/bootstrap/precedence/docs.png)
![Precedence issue-close-require.yml](twbs/bootstrap/precedence/issue-close-require.png)
![Precedence issue-labeled.yml](twbs/bootstrap/precedence/issue-labeled.png)
![Precedence js.yml](twbs/bootstrap/precedence/js.png)
![Precedence lint.yml](twbs/bootstrap/precedence/lint.png)
![Precedence node-sass.yml](twbs/bootstrap/precedence/node-sass.png)
![Precedence release-notes.yml](twbs/bootstrap/precedence/release-notes.png)

#### Dependencies

![Dependencies browserstack.yml](twbs/bootstrap/dependencies/browserstack.png)
![Dependencies bundlewatch.yml](twbs/bootstrap/dependencies/bundlewatch.png)
![Dependencies calibreapp-image-actions.yml](twbs/bootstrap/dependencies/calibreapp-image-actions.png)
![Dependencies codeql.yml](twbs/bootstrap/dependencies/codeql.png)
![Dependencies cspell.yml](twbs/bootstrap/dependencies/cspell.png)
![Dependencies css.yml](twbs/bootstrap/dependencies/css.png)
![Dependencies docs.yml](twbs/bootstrap/dependencies/docs.png)
![Dependencies issue-close-require.yml](twbs/bootstrap/dependencies/issue-close-require.png)
![Dependencies issue-labeled.yml](twbs/bootstrap/dependencies/issue-labeled.png)
![Dependencies js.yml](twbs/bootstrap/dependencies/js.png)
![Dependencies lint.yml](twbs/bootstrap/dependencies/lint.png)
![Dependencies node-sass.yml](twbs/bootstrap/dependencies/node-sass.png)
![Dependencies release-notes.yml](twbs/bootstrap/dependencies/release-notes.png)

### ytdl-org/youtube-dl

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name             | Up to date |
|-------------|-------------------------|------------|
| GITHUB      | actions/checkout@v2     | False      |
| GITHUB      | actions/setup-python@v2 | False      |
| GITHUB      | actions/setup-java@v1   | False      |

#### Precedence

![Precedence ci.yml](ytdl-org/youtube-dl/precedence/ci.png)

#### Dependencies

![Dependencies ci.yml](ytdl-org/youtube-dl/dependencies/ci.png)

### vuejs/vue

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.29%      | 1                 |
| TRUSTED     | 0.29%      | 1                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                  | Up to date |
|-------------|------------------------------|------------|
| GITHUB      | actions/checkout@v2          | False      |
| GITHUB      | actions/setup-node@v2        | False      |
| PUBLIC      | pnpm/action-setup@v2         | True       |
| GITHUB      | actions/checkout@master      | False      |
| TRUSTED     | yyx990803/release-tag@master | False      |

#### Precedence

![Precedence ci.yml](vuejs/vue/precedence/ci.png)
![Precedence release-tag.yml](vuejs/vue/precedence/release-tag.png)

#### Dependencies

![Dependencies ci.yml](vuejs/vue/dependencies/ci.png)
![Dependencies release-tag.yml](vuejs/vue/dependencies/release-tag.png)

### vuejs/docs

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.0%       | 0                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.29%      | 1                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                      | Up to date |
|-------------|----------------------------------|------------|
| PUBLIC      | dependabot/fetch-metadata@v1.1.1 | False      |

#### Precedence

![Precedence automerge.yml](vuejs/docs/precedence/automerge.png)

#### Dependencies

![Dependencies automerge.yml](vuejs/docs/dependencies/automerge.png)

### vuejs/router

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.29%      | 1                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                                  | Up to date |
|-------------|----------------------------------------------|------------|
| GITHUB      | actions/checkout@master                      | False      |
| TRUSTED     | yyx990803/release-tag@master                 | False      |
| GITHUB      | actions/checkout@v3                          | True       |
| GITHUB      | actions/setup-node@v2                        | False      |
| PUBLIC      | pnpm/action-setup@v2.2.1                     | False      |
| PUBLIC      | browserstack/github-actions/setup-env@master | False      |
| PUBLIC      | codecov/codecov-action@v2                    | False      |

#### Precedence

![Precedence release-tag.yml](vuejs/router/precedence/release-tag.png)
![Precedence test.yml](vuejs/router/precedence/test.png)

#### Dependencies

![Dependencies release-tag.yml](vuejs/router/dependencies/release-tag.png)
![Dependencies test.yml](vuejs/router/dependencies/test.png)

### microsoft/FluidFramework

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 3.54%      | 12                |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 3.83%      | 13                |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                                                                         | Up to date |
|-------------|-------------------------------------------------------------------------------------|------------|
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c                           | False      |
| PUBLIC      | github/codeql-action/init@a34ca99b4610d924e04c68db79e503e1f79f9f02                  | False      |
| PUBLIC      | github/codeql-action/autobuild@a34ca99b4610d924e04c68db79e503e1f79f9f02             | False      |
| PUBLIC      | github/codeql-action/analyze@a34ca99b4610d924e04c68db79e503e1f79f9f02               | False      |
| PUBLIC      | dawidd6/action-download-artifact@bd10f381a96414ce2b13a11bfa89902ba7cea07f           | False      |
| PUBLIC      | marocchino/sticky-pull-request-comment@fcf6fe9e4a0409cd9316a5011435be0f3327f1e1     | False      |
| GITHUB      | actions/github-script@d556feaca394842dc55e4734bf3bb9f685482fa0                      | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c                           | False      |
| GITHUB      | actions/setup-node@64ed1c7eab4cce3362f8c340dee64e5eaeef8f7c                         | False      |
| GITHUB      | actions/labeler@5c7539237e04b714afd8ad9b4aed733815b9fab4                            | False      |
| PUBLIC      | srvaroa/labeler@97fabbad5804e8a22d5f027aa94c98614facb571                            | False      |
| PUBLIC      | tylerbutler/labelmaker-action@49487085eebc5be6b766198e231f0688e4b4a7c2              | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c                           | False      |
| PUBLIC      | dorny/paths-filter@4512585405083f25c027a35db413c2b3b9006d50                         | False      |
| PUBLIC      | mszostok/codeowners-validator@7f3f5e28c6d7b8dfae5731e54ce2272ca384592f              | False      |
| PUBLIC      | sitezen/pr-comment-checker@f1e956fac00c6d1163d15841886ae80b7ae58ecb                 | False      |
| PUBLIC      | pnpm/action-setup@c3b53f6a16e57305370b4ae5a540c2077a1d50dd                          | False      |
| PUBLIC      | JulienKode/pull-request-name-linter-action@8c05fb989d9f156ce61e33754f9802c9d3cffa58 | False      |
| GITHUB      | actions/github-script@d556feaca394842dc55e4734bf3bb9f685482fa0                      | False      |
| GITHUB      | actions/checkout@7884fcad6b5d53d10323aee724dc68d8b9096a2e                           | False      |
| PUBLIC      | beatlabs/delete-old-branches-action@db61ade054731e37b5740e23336445fbc75ccd7b        | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c                           | False      |
| GITHUB      | actions/setup-node@64ed1c7eab4cce3362f8c340dee64e5eaeef8f7c                         | False      |
| GITHUB      | actions/upload-artifact@0b7f8abb1508181956e8e162db84b466c27e18ce                    | False      |
| GITHUB      | actions/download-artifact@9bc31d5ccc31df68ecc42ccf4149144866c47d8a                  | False      |

#### Precedence

![Precedence codeql-analysis.yml](microsoft/FluidFramework/precedence/codeql-analysis.png)
![Precedence linkcheck-reporter.yml](microsoft/FluidFramework/precedence/linkcheck-reporter.png)
![Precedence merge-commits.yml](microsoft/FluidFramework/precedence/merge-commits.png)
![Precedence pr-labeler.yml](microsoft/FluidFramework/precedence/pr-labeler.png)
![Precedence pr-validation.yml](microsoft/FluidFramework/precedence/pr-validation.png)
![Precedence push-queue.yml](microsoft/FluidFramework/precedence/push-queue.png)
![Precedence stale-branches.yml](microsoft/FluidFramework/precedence/stale-branches.png)
![Precedence website-validation.yml](microsoft/FluidFramework/precedence/website-validation.png)

#### Dependencies

![Dependencies codeql-analysis.yml](microsoft/FluidFramework/dependencies/codeql-analysis.png)
![Dependencies linkcheck-reporter.yml](microsoft/FluidFramework/dependencies/linkcheck-reporter.png)
![Dependencies merge-commits.yml](microsoft/FluidFramework/dependencies/merge-commits.png)
![Dependencies pr-labeler.yml](microsoft/FluidFramework/dependencies/pr-labeler.png)
![Dependencies pr-validation.yml](microsoft/FluidFramework/dependencies/pr-validation.png)
![Dependencies push-queue.yml](microsoft/FluidFramework/dependencies/push-queue.png)
![Dependencies stale-branches.yml](microsoft/FluidFramework/dependencies/stale-branches.png)
![Dependencies website-validation.yml](microsoft/FluidFramework/dependencies/website-validation.png)

### microsoft/TypeScript

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 2.65%      | 9                 |
| INTERNAL    | 0.29%      | 1                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                       | Up to date |
|-------------|-----------------------------------|------------|
| GITHUB      | actions/checkout@v3               | True       |
| GITHUB      | actions/setup-node@v3             | True       |
| GITHUB      | actions/checkout@v3               | True       |
| PUBLIC      | github/codeql-action/init@v2      | False      |
| PUBLIC      | github/codeql-action/autobuild@v2 | False      |
| PUBLIC      | github/codeql-action/analyze@v2   | False      |
| GITHUB      | actions/checkout@v3               | True       |
| GITHUB      | actions/checkout@v3               | True       |
| GITHUB      | actions/setup-node@v3             | True       |
| INTERNAL    | microsoft/RichCodeNavIndexer@v0.1 | False      |
| GITHUB      | actions/checkout@v3               | True       |
| GITHUB      | actions/checkout@v3               | True       |
| GITHUB      | actions/setup-node@v3             | True       |

#### Precedence

![Precedence ci.yml](microsoft/TypeScript/precedence/ci.png)
![Precedence codeql.yml](microsoft/TypeScript/precedence/codeql.png)
![Precedence ensure-related-repos-run-crons.yml](microsoft/TypeScript/precedence/ensure-related-repos-run-crons.png)
![Precedence rich-navigation.yml](microsoft/TypeScript/precedence/rich-navigation.png)
![Precedence sync-wiki.yml](microsoft/TypeScript/precedence/sync-wiki.png)
![Precedence update-lkg.yml](microsoft/TypeScript/precedence/update-lkg.png)

#### Dependencies

![Dependencies ci.yml](microsoft/TypeScript/dependencies/ci.png)
![Dependencies codeql.yml](microsoft/TypeScript/dependencies/codeql.png)
![Dependencies ensure-related-repos-run-crons.yml](microsoft/TypeScript/dependencies/ensure-related-repos-run-crons.png)
![Dependencies rich-navigation.yml](microsoft/TypeScript/dependencies/rich-navigation.png)
![Dependencies sync-wiki.yml](microsoft/TypeScript/dependencies/sync-wiki.png)
![Dependencies update-lkg.yml](microsoft/TypeScript/dependencies/update-lkg.png)

### microsoft/winget-cli

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.0%       | 0                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.29%      | 1                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                           | Up to date |
|-------------|---------------------------------------|------------|
| PUBLIC      | check-spelling/check-spelling@v0.0.21 | True       |

#### Precedence

![Precedence spelling.yml](microsoft/winget-cli/precedence/spelling.png)
![Precedence spelling2.yml](microsoft/winget-cli/precedence/spelling2.png)
![Precedence spelling3.yml](microsoft/winget-cli/precedence/spelling3.png)

#### Dependencies

![Dependencies spelling.yml](microsoft/winget-cli/dependencies/spelling.png)
![Dependencies spelling2.yml](microsoft/winget-cli/dependencies/spelling2.png)
![Dependencies spelling3.yml](microsoft/winget-cli/dependencies/spelling3.png)

### microsoft/fluentui-react-native

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.29%      | 1                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                       | Up to date |
|-------------|-----------------------------------|------------|
| GITHUB      | actions/checkout@v2               | False      |
| PUBLIC      | github/codeql-action/init@v1      | False      |
| PUBLIC      | github/codeql-action/autobuild@v1 | False      |
| PUBLIC      | github/codeql-action/analyze@v1   | False      |

#### Precedence

![Precedence codeql-analysis.yml](microsoft/fluentui-react-native/precedence/codeql-analysis.png)

#### Dependencies

![Dependencies codeql-analysis.yml](microsoft/fluentui-react-native/dependencies/codeql-analysis.png)

### microsoft/azuredatastudio

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 2.06%      | 7                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.59%      | 2                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                       | Up to date |
|-------------|-----------------------------------|------------|
| GITHUB      | actions/checkout@v2               | False      |
| GITHUB      | actions/setup-node@v2             | False      |
| GITHUB      | actions/setup-python@v2           | False      |
| GITHUB      | actions/checkout@v2.2.0           | False      |
| GITHUB      | actions/cache@v2                  | False      |
| PUBLIC      | coverallsapp/github-action@v1.1.1 | False      |
| GITHUB      | actions/checkout@v2               | False      |
| PUBLIC      | hramos/label-actions@v1           | False      |
| GITHUB      | actions/checkout@v2               | False      |

#### Precedence

![Precedence ci.yml](microsoft/azuredatastudio/precedence/ci.png)
![Precedence on-label.yml](microsoft/azuredatastudio/precedence/on-label.png)
![Precedence on-pr-open.yml](microsoft/azuredatastudio/precedence/on-pr-open.png)

#### Dependencies

![Dependencies ci.yml](microsoft/azuredatastudio/dependencies/ci.png)
![Dependencies on-label.yml](microsoft/azuredatastudio/dependencies/on-label.png)
![Dependencies on-pr-open.yml](microsoft/azuredatastudio/dependencies/on-pr-open.png)

### microsoft/vscode

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 9.73%      | 33                |
| INTERNAL    | 0.29%      | 1                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                                                         | Up to date |
|-------------|---------------------------------------------------------------------|------------|
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/setup-node@v3                                               | True       |
| GITHUB      | actions/cache@v3                                                    | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/setup-node@v3                                               | True       |
| GITHUB      | actions/setup-python@v4                                             | True       |
| GITHUB      | actions/cache@v3                                                    | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/setup-python@v4                                             | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| PUBLIC      | azure/login@92a5484dfaf04ca78a94597f4f19fea633851fa2                | False      |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/setup-node@v3                                               | True       |
| GITHUB      | actions/cache@v3                                                    | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| PUBLIC      | octokit/request-action@v2.x                                         | False      |
| PUBLIC      | trilom/file-changes-action@ce38c8ce2459ca3c303415eec8cb0409857b4272 | False      |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/cache@v3                                                    | True       |
| GITHUB      | actions/setup-node@v3                                               | True       |
| INTERNAL    | microsoft/RichCodeNavIndexer@v0.1                                   | False      |
| GITHUB      | actions/checkout@v3                                                 | True       |
| GITHUB      | actions/setup-node@v3                                               | True       |
| GITHUB      | actions/checkout@v3                                                 | True       |

#### Precedence

![Precedence author-verified.yml](microsoft/vscode/precedence/author-verified.png)
![Precedence bad-tag.yml](microsoft/vscode/precedence/bad-tag.png)
![Precedence basic.yml](microsoft/vscode/precedence/basic.png)
![Precedence ci.yml](microsoft/vscode/precedence/ci.png)
![Precedence deep-classifier-assign-monitor.yml](microsoft/vscode/precedence/deep-classifier-assign-monitor.png)
![Precedence deep-classifier-runner.yml](microsoft/vscode/precedence/deep-classifier-runner.png)
![Precedence deep-classifier-scraper.yml](microsoft/vscode/precedence/deep-classifier-scraper.png)
![Precedence deep-classifier-unassign-monitor.yml](microsoft/vscode/precedence/deep-classifier-unassign-monitor.png)
![Precedence devcontainer-cache.yml](microsoft/vscode/precedence/devcontainer-cache.png)
![Precedence english-please.yml](microsoft/vscode/precedence/english-please.png)
![Precedence feature-request.yml](microsoft/vscode/precedence/feature-request.png)
![Precedence latest-release-monitor.yml](microsoft/vscode/precedence/latest-release-monitor.png)
![Precedence locker.yml](microsoft/vscode/precedence/locker.png)
![Precedence monaco-editor.yml](microsoft/vscode/precedence/monaco-editor.png)
![Precedence needs-more-info-closer.yml](microsoft/vscode/precedence/needs-more-info-closer.png)
![Precedence no-yarn-lock-changes.yml](microsoft/vscode/precedence/no-yarn-lock-changes.png)
![Precedence on-comment.yml](microsoft/vscode/precedence/on-comment.png)
![Precedence on-label.yml](microsoft/vscode/precedence/on-label.png)
![Precedence on-open.yml](microsoft/vscode/precedence/on-open.png)
![Precedence release-pipeline-labeler.yml](microsoft/vscode/precedence/release-pipeline-labeler.png)
![Precedence rich-navigation.yml](microsoft/vscode/precedence/rich-navigation.png)
![Precedence telemetry.yml](microsoft/vscode/precedence/telemetry.png)
![Precedence test-plan-item-validator.yml](microsoft/vscode/precedence/test-plan-item-validator.png)

#### Dependencies

![Dependencies author-verified.yml](microsoft/vscode/dependencies/author-verified.png)
![Dependencies bad-tag.yml](microsoft/vscode/dependencies/bad-tag.png)
![Dependencies basic.yml](microsoft/vscode/dependencies/basic.png)
![Dependencies ci.yml](microsoft/vscode/dependencies/ci.png)
![Dependencies deep-classifier-assign-monitor.yml](microsoft/vscode/dependencies/deep-classifier-assign-monitor.png)
![Dependencies deep-classifier-runner.yml](microsoft/vscode/dependencies/deep-classifier-runner.png)
![Dependencies deep-classifier-scraper.yml](microsoft/vscode/dependencies/deep-classifier-scraper.png)
![Dependencies deep-classifier-unassign-monitor.yml](microsoft/vscode/dependencies/deep-classifier-unassign-monitor.png)
![Dependencies devcontainer-cache.yml](microsoft/vscode/dependencies/devcontainer-cache.png)
![Dependencies english-please.yml](microsoft/vscode/dependencies/english-please.png)
![Dependencies feature-request.yml](microsoft/vscode/dependencies/feature-request.png)
![Dependencies latest-release-monitor.yml](microsoft/vscode/dependencies/latest-release-monitor.png)
![Dependencies locker.yml](microsoft/vscode/dependencies/locker.png)
![Dependencies monaco-editor.yml](microsoft/vscode/dependencies/monaco-editor.png)
![Dependencies needs-more-info-closer.yml](microsoft/vscode/dependencies/needs-more-info-closer.png)
![Dependencies no-yarn-lock-changes.yml](microsoft/vscode/dependencies/no-yarn-lock-changes.png)
![Dependencies on-comment.yml](microsoft/vscode/dependencies/on-comment.png)
![Dependencies on-label.yml](microsoft/vscode/dependencies/on-label.png)
![Dependencies on-open.yml](microsoft/vscode/dependencies/on-open.png)
![Dependencies release-pipeline-labeler.yml](microsoft/vscode/dependencies/release-pipeline-labeler.png)
![Dependencies rich-navigation.yml](microsoft/vscode/dependencies/rich-navigation.png)
![Dependencies telemetry.yml](microsoft/vscode/dependencies/telemetry.png)
![Dependencies test-plan-item-validator.yml](microsoft/vscode/dependencies/test-plan-item-validator.png)

### collet/cucumber-demo

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.59%      | 2                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name           | Up to date |
|-------------|-----------------------|------------|
| GITHUB      | actions/checkout@v2   | False      |
| GITHUB      | actions/setup-java@v1 | False      |

#### Precedence

![Precedence maven.yml](collet/cucumber-demo/precedence/maven.png)

#### Dependencies

![Dependencies maven.yml](collet/cucumber-demo/dependencies/maven.png)

### mathiascouste/qgl-template

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.59%      | 2                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name           | Up to date |
|-------------|-----------------------|------------|
| GITHUB      | actions/checkout@v2   | False      |
| GITHUB      | actions/setup-java@v1 | False      |

#### Precedence

![Precedence pr-build.yml](mathiascouste/qgl-template/precedence/pr-build.png)

#### Dependencies

![Dependencies pr-build.yml](mathiascouste/qgl-template/dependencies/pr-build.png)

### vitest-dev/vitest

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 1.47%      | 5                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                   | Up to date |
|-------------|-------------------------------|------------|
| GITHUB      | actions/checkout@v3           | True       |
| GITHUB      | actions/setup-node@v3         | True       |
| PUBLIC      | pnpm/action-setup@v2          | True       |
| GITHUB      | actions/checkout@v3           | True       |
| PUBLIC      | actions-cool/issues-helper@v3 | True       |
| PUBLIC      | actions-cool/issues-helper@v3 | True       |
| GITHUB      | actions/checkout@v3           | True       |
| GITHUB      | actions/setup-node@v3         | True       |

#### Precedence

![Precedence bench.yml](vitest-dev/vitest/precedence/bench.png)
![Precedence ci.yml](vitest-dev/vitest/precedence/ci.png)
![Precedence issue-close-require.yml](vitest-dev/vitest/precedence/issue-close-require.png)
![Precedence issue-labeled.yml](vitest-dev/vitest/precedence/issue-labeled.png)
![Precedence release.yml](vitest-dev/vitest/precedence/release.png)

#### Dependencies

![Dependencies bench.yml](vitest-dev/vitest/dependencies/bench.png)
![Dependencies ci.yml](vitest-dev/vitest/dependencies/ci.png)
![Dependencies issue-close-require.yml](vitest-dev/vitest/dependencies/issue-close-require.png)
![Dependencies issue-labeled.yml](vitest-dev/vitest/dependencies/issue-labeled.png)
![Dependencies release.yml](vitest-dev/vitest/dependencies/release.png)

### i18next/next-i18next

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.59%      | 2                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name           | Up to date |
|-------------|-----------------------|------------|
| GITHUB      | actions/checkout@v3   | True       |
| GITHUB      | actions/setup-node@v3 | True       |

#### Precedence

![Precedence ci.yml](i18next/next-i18next/precedence/ci.png)

#### Dependencies

![Dependencies ci.yml](i18next/next-i18next/dependencies/ci.png)

### jwasham/coding-interview-university

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.29%      | 1                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 1.18%      | 4                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                           | Up to date |
|-------------|---------------------------------------|------------|
| GITHUB      | actions/checkout@v3                   | True       |
| PUBLIC      | lycheeverse/lychee-action@v1.4.1      | False      |
| PUBLIC      | micalevisk/last-issue-action@v1.2     | False      |
| PUBLIC      | peter-evans/create-issue-from-file@v4 | True       |
| PUBLIC      | peter-evans/close-issue@v2            | True       |

#### Precedence

![Precedence links_checker.yml](jwasham/coding-interview-university/precedence/links_checker.png)

#### Dependencies

![Dependencies links_checker.yml](jwasham/coding-interview-university/dependencies/links_checker.png)

### EbookFoundation/free-programming-books

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 1.47%      | 5                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 1.47%      | 5                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                                 | Up to date |
|-------------|---------------------------------------------|------------|
| GITHUB      | actions/checkout@v3                         | True       |
| GITHUB      | actions/upload-artifact@v3                  | True       |
| PUBLIC      | tj-actions/changed-files@v35.4.4            | False      |
| PUBLIC      | ruby/setup-ruby@v1                          | True       |
| PUBLIC      | eps1lon/actions-label-merge-conflict@v2.1.0 | True       |
| GITHUB      | actions/checkout@v3                         | True       |
| GITHUB      | actions/setup-node@v3                       | True       |
| PUBLIC      | actions-ecosystem/action-add-labels@v1      | True       |
| PUBLIC      | actions-ecosystem/action-remove-labels@v1   | True       |
| GITHUB      | actions/stale@v7                            | True       |

#### Precedence

![Precedence check-urls.yml](EbookFoundation/free-programming-books/precedence/check-urls.png)
![Precedence detect-conflicting-prs.yml](EbookFoundation/free-programming-books/precedence/detect-conflicting-prs.png)
![Precedence fpb-lint.yml](EbookFoundation/free-programming-books/precedence/fpb-lint.png)
![Precedence issues-pinner.yml](EbookFoundation/free-programming-books/precedence/issues-pinner.png)
![Precedence stale.yml](EbookFoundation/free-programming-books/precedence/stale.png)

#### Dependencies

![Dependencies check-urls.yml](EbookFoundation/free-programming-books/dependencies/check-urls.png)
![Dependencies detect-conflicting-prs.yml](EbookFoundation/free-programming-books/dependencies/detect-conflicting-prs.png)
![Dependencies fpb-lint.yml](EbookFoundation/free-programming-books/dependencies/fpb-lint.png)
![Dependencies issues-pinner.yml](EbookFoundation/free-programming-books/dependencies/issues-pinner.png)
![Dependencies stale.yml](EbookFoundation/free-programming-books/dependencies/stale.png)

### flutter/flutter

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 1.18%      | 4                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                                                                | Up to date |
|-------------|----------------------------------------------------------------------------|------------|
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c                  | False      |
| PUBLIC      | codecov/codecov-action@d9f34f8cd5cb3b3eb79b3e4b5dae3a16df499a70            | False      |
| PUBLIC      | google/mirror-branch-action@c6b07e441a7ffc5ae15860c1d0a8107a3a151db8       | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c                  | False      |
| GITHUB      | actions/upload-artifact@0b7f8abb1508181956e8e162db84b466c27e18ce           | False      |
| PUBLIC      | ossf/scorecard-action@e38b1902ae4f44df626f11ba0734b14fb91f8f86             | False      |
| PUBLIC      | github/codeql-action/upload-sarif@a34ca99b4610d924e04c68db79e503e1f79f9f02 | False      |

#### Precedence

![Precedence coverage.yml](flutter/flutter/precedence/coverage.png)
![Precedence mirror.yml](flutter/flutter/precedence/mirror.png)
![Precedence scorecards-analysis.yml](flutter/flutter/precedence/scorecards-analysis.png)

#### Dependencies

![Dependencies coverage.yml](flutter/flutter/dependencies/coverage.png)
![Dependencies mirror.yml](flutter/flutter/dependencies/mirror.png)
![Dependencies scorecards-analysis.yml](flutter/flutter/dependencies/scorecards-analysis.png)

### mobileandyou/api

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                  | Up to date |
|-------------|------------------------------|------------|
| GITHUB      | actions/checkout@v2          | False      |
| GITHUB      | actions/setup-java@v1        | False      |
| PUBLIC      | appleboy/scp-action@master   | False      |
| PUBLIC      | appleboy/ssh-action@master   | False      |
| GITHUB      | actions/checkout@v3          | True       |
| PUBLIC      | JetBrains/qodana-action@main | False      |

#### Precedence

![Precedence api.yml](mobileandyou/api/precedence/api.png)
![Precedence qodana.yml](mobileandyou/api/precedence/qodana.png)

#### Dependencies

![Dependencies api.yml](mobileandyou/api/dependencies/api.png)
![Dependencies qodana.yml](mobileandyou/api/dependencies/qodana.png)

### facebook/react

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 1.77%      | 6                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.29%      | 1                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                             | Up to date |
|-------------|-----------------------------------------|------------|
| GITHUB      | actions/setup-node@v3                   | True       |
| GITHUB      | actions/github-script@v6                | True       |
| GITHUB      | actions/upload-artifact@v3              | True       |
| GITHUB      | actions/checkout@v3                     | True       |
| GITHUB      | actions/download-artifact@v3            | True       |
| PUBLIC      | stefanzweifel/git-auto-commit-action@v4 | True       |
| GITHUB      | actions/github-script@v3                | False      |

#### Precedence

![Precedence commit_artifacts.yml](facebook/react/precedence/commit_artifacts.png)
![Precedence devtools_check_repro.yml](facebook/react/precedence/devtools_check_repro.png)

#### Dependencies

![Dependencies commit_artifacts.yml](facebook/react/dependencies/commit_artifacts.png)
![Dependencies devtools_check_repro.yml](facebook/react/dependencies/devtools_check_repro.png)

### freeCodeCamp/freeCodeCamp

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 8.85%      | 30                |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 5.6%       | 19                |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                                                           | Up to date |
|-------------|-----------------------------------------------------------------------|------------|
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| PUBLIC      | github/codeql-action/init@3ebbd71c74ef574dbc558c82f70e52732c8b44fe    | False      |
| PUBLIC      | github/codeql-action/analyze@3ebbd71c74ef574dbc558c82f70e52732c8b44fe | False      |
| PUBLIC      | Codesee-io/codesee-action@1d109bb07bbd63a6fc3d01b40d28a4c8f0925bf5    | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| PUBLIC      | freecodecamp/crowdin-action@main                                      | False      |
| PUBLIC      | crowdin/github-action@master                                          | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| GITHUB      | actions/setup-node@64ed1c7eab4cce3362f8c340dee64e5eaeef8f7c           | False      |
| PUBLIC      | freecodecamp/crowdin-action@main                                      | False      |
| PUBLIC      | crowdin/github-action@master                                          | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| PUBLIC      | freecodecamp/crowdin-action@main                                      | False      |
| PUBLIC      | crowdin/github-action@master                                          | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| PUBLIC      | freecodecamp/crowdin-action@main                                      | False      |
| PUBLIC      | crowdin/github-action@master                                          | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| PUBLIC      | freecodecamp/crowdin-action@main                                      | False      |
| PUBLIC      | crowdin/github-action@master                                          | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| PUBLIC      | freecodecamp/crowdin-action@main                                      | False      |
| PUBLIC      | crowdin/github-action@master                                          | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| GITHUB      | actions/setup-node@64ed1c7eab4cce3362f8c340dee64e5eaeef8f7c           | False      |
| PUBLIC      | subosito/flutter-action@dbf1fa04f4d2e52c33185153d06cdb5443aa189d      | False      |
| PUBLIC      | cypress-io/github-action@v4                                           | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| GITHUB      | actions/setup-node@64ed1c7eab4cce3362f8c340dee64e5eaeef8f7c           | False      |
| PUBLIC      | cypress-io/github-action@v4                                           | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| GITHUB      | actions/setup-node@64ed1c7eab4cce3362f8c340dee64e5eaeef8f7c           | False      |
| GITHUB      | actions/upload-artifact@0b7f8abb1508181956e8e162db84b466c27e18ce      | False      |
| GITHUB      | actions/download-artifact@9bc31d5ccc31df68ecc42ccf4149144866c47d8a    | False      |
| PUBLIC      | cypress-io/github-action@v4                                           | False      |
| GITHUB      | actions/github-script@d556feaca394842dc55e4734bf3bb9f685482fa0        | False      |
| GITHUB      | actions/github-script@d556feaca394842dc55e4734bf3bb9f685482fa0        | False      |
| GITHUB      | actions/github-script@d556feaca394842dc55e4734bf3bb9f685482fa0        | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| GITHUB      | actions/setup-node@64ed1c7eab4cce3362f8c340dee64e5eaeef8f7c           | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| GITHUB      | actions/setup-node@64ed1c7eab4cce3362f8c340dee64e5eaeef8f7c           | False      |
| GITHUB      | actions/github-script@d556feaca394842dc55e4734bf3bb9f685482fa0        | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| GITHUB      | actions/setup-node@64ed1c7eab4cce3362f8c340dee64e5eaeef8f7c           | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| GITHUB      | actions/setup-node@64ed1c7eab4cce3362f8c340dee64e5eaeef8f7c           | False      |
| GITHUB      | actions/checkout@ac593985615ec2ede58e132d2e21d2b1cbd6127c             | False      |
| GITHUB      | actions/setup-node@64ed1c7eab4cce3362f8c340dee64e5eaeef8f7c           | False      |

#### Precedence

![Precedence codeql-analysis.yml](freeCodeCamp/freeCodeCamp/precedence/codeql-analysis.png)
![Precedence codesee-diagram.yml](freeCodeCamp/freeCodeCamp/precedence/codesee-diagram.png)
![Precedence crowdin-download.client-ui.yml](freeCodeCamp/freeCodeCamp/precedence/crowdin-download.client-ui.png)
![Precedence crowdin-download.curriculum.yml](freeCodeCamp/freeCodeCamp/precedence/crowdin-download.curriculum.png)
![Precedence crowdin-download.docs.yml](freeCodeCamp/freeCodeCamp/precedence/crowdin-download.docs.png)
![Precedence crowdin-upload.client-ui.yml](freeCodeCamp/freeCodeCamp/precedence/crowdin-upload.client-ui.png)
![Precedence crowdin-upload.curriculum.yml](freeCodeCamp/freeCodeCamp/precedence/crowdin-upload.curriculum.png)
![Precedence crowdin-upload.docs.yml](freeCodeCamp/freeCodeCamp/precedence/crowdin-upload.docs.png)
![Precedence e2e-mobile.yml](freeCodeCamp/freeCodeCamp/precedence/e2e-mobile.png)
![Precedence e2e-third-party.yml](freeCodeCamp/freeCodeCamp/precedence/e2e-third-party.png)
![Precedence e2e-web.yml](freeCodeCamp/freeCodeCamp/precedence/e2e-web.png)
![Precedence github-autoclose.yml](freeCodeCamp/freeCodeCamp/precedence/github-autoclose.png)
![Precedence github-no-i18n-via-prs.yml](freeCodeCamp/freeCodeCamp/precedence/github-no-i18n-via-prs.png)
![Precedence github-spam.yml](freeCodeCamp/freeCodeCamp/precedence/github-spam.png)
![Precedence i18n-validate-builds.yml](freeCodeCamp/freeCodeCamp/precedence/i18n-validate-builds.png)
![Precedence i18n-validate-prs.yml](freeCodeCamp/freeCodeCamp/precedence/i18n-validate-prs.png)
![Precedence node.js-find-unused.yml](freeCodeCamp/freeCodeCamp/precedence/node.js-find-unused.png)
![Precedence node.js-tests-upcoming.yml](freeCodeCamp/freeCodeCamp/precedence/node.js-tests-upcoming.png)
![Precedence node.js-tests.yml](freeCodeCamp/freeCodeCamp/precedence/node.js-tests.png)

#### Dependencies

![Dependencies codeql-analysis.yml](freeCodeCamp/freeCodeCamp/dependencies/codeql-analysis.png)
![Dependencies codesee-diagram.yml](freeCodeCamp/freeCodeCamp/dependencies/codesee-diagram.png)
![Dependencies crowdin-download.client-ui.yml](freeCodeCamp/freeCodeCamp/dependencies/crowdin-download.client-ui.png)
![Dependencies crowdin-download.curriculum.yml](freeCodeCamp/freeCodeCamp/dependencies/crowdin-download.curriculum.png)
![Dependencies crowdin-download.docs.yml](freeCodeCamp/freeCodeCamp/dependencies/crowdin-download.docs.png)
![Dependencies crowdin-upload.client-ui.yml](freeCodeCamp/freeCodeCamp/dependencies/crowdin-upload.client-ui.png)
![Dependencies crowdin-upload.curriculum.yml](freeCodeCamp/freeCodeCamp/dependencies/crowdin-upload.curriculum.png)
![Dependencies crowdin-upload.docs.yml](freeCodeCamp/freeCodeCamp/dependencies/crowdin-upload.docs.png)
![Dependencies e2e-mobile.yml](freeCodeCamp/freeCodeCamp/dependencies/e2e-mobile.png)
![Dependencies e2e-third-party.yml](freeCodeCamp/freeCodeCamp/dependencies/e2e-third-party.png)
![Dependencies e2e-web.yml](freeCodeCamp/freeCodeCamp/dependencies/e2e-web.png)
![Dependencies github-autoclose.yml](freeCodeCamp/freeCodeCamp/dependencies/github-autoclose.png)
![Dependencies github-no-i18n-via-prs.yml](freeCodeCamp/freeCodeCamp/dependencies/github-no-i18n-via-prs.png)
![Dependencies github-spam.yml](freeCodeCamp/freeCodeCamp/dependencies/github-spam.png)
![Dependencies i18n-validate-builds.yml](freeCodeCamp/freeCodeCamp/dependencies/i18n-validate-builds.png)
![Dependencies i18n-validate-prs.yml](freeCodeCamp/freeCodeCamp/dependencies/i18n-validate-prs.png)
![Dependencies node.js-find-unused.yml](freeCodeCamp/freeCodeCamp/dependencies/node.js-find-unused.png)
![Dependencies node.js-tests-upcoming.yml](freeCodeCamp/freeCodeCamp/dependencies/node.js-tests-upcoming.png)
![Dependencies node.js-tests.yml](freeCodeCamp/freeCodeCamp/dependencies/node.js-tests.png)

### d3/d3

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.59%      | 2                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name           | Up to date |
|-------------|-----------------------|------------|
| GITHUB      | actions/checkout@v2   | False      |
| GITHUB      | actions/setup-node@v1 | False      |

#### Precedence

![Precedence node.js.yml](d3/d3/precedence/node.js.png)

#### Dependencies

![Dependencies node.js.yml](d3/d3/dependencies/node.js.png)

### mui/material-ui

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 1.18%      | 4                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 2.65%      | 9                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                                                                     | Up to date |
|-------------|---------------------------------------------------------------------------------|------------|
| PUBLIC      | mnajdova/github-action-required-labels@ca0df9249827e43aa4b4a0d25d9fe3e9b19b0705 | False      |
| GITHUB      | actions/checkout@755da8c3cf115ac066823e79a1e1788f8940201b                       | False      |
| GITHUB      | actions/setup-node@8c91899e586c5b171469028077307d293428b516                     | False      |
| GITHUB      | actions/checkout@755da8c3cf115ac066823e79a1e1788f8940201b                       | False      |
| PUBLIC      | github/codeql-action/init@959cbb7472c4d4ad70cdfe6f4976053fe48ab394              | False      |
| PUBLIC      | github/codeql-action/analyze@959cbb7472c4d4ad70cdfe6f4976053fe48ab394           | False      |
| PUBLIC      | eps1lon/actions-label-merge-conflict@fd1f295ee7443d13745804bc49fe158e240f6c6e   | False      |
| PUBLIC      | actions-cool/issues-helper@275328970dbc3bfc3bc43f5fe741bf3638300c0a             | False      |
| PUBLIC      | lee-dohm/no-response@9bb0a4b5e6a45046f00353d5de7d90fb8bd773bb                   | False      |
| GITHUB      | actions/checkout@755da8c3cf115ac066823e79a1e1788f8940201b                       | False      |
| PUBLIC      | ossf/scorecard-action@e38b1902ae4f44df626f11ba0734b14fb91f8f86                  | False      |
| PUBLIC      | github/codeql-action/upload-sarif@959cbb7472c4d4ad70cdfe6f4976053fe48ab394      | False      |
| PUBLIC      | dessant/support-requests@b1303caf4438e66dea1130aa4c30189dc28e690d               | False      |

#### Precedence

![Precedence check-if-pr-has-label.yml](mui/material-ui/precedence/check-if-pr-has-label.png)
![Precedence ci-check.yml](mui/material-ui/precedence/ci-check.png)
![Precedence ci.yml](mui/material-ui/precedence/ci.png)
![Precedence codeql.yml](mui/material-ui/precedence/codeql.png)
![Precedence maintenance.yml](mui/material-ui/precedence/maintenance.png)
![Precedence mark-duplicate.yml](mui/material-ui/precedence/mark-duplicate.png)
![Precedence no-response.yml](mui/material-ui/precedence/no-response.png)
![Precedence scorecards.yml](mui/material-ui/precedence/scorecards.png)
![Precedence support-stackoverflow.yml](mui/material-ui/precedence/support-stackoverflow.png)

#### Dependencies

![Dependencies check-if-pr-has-label.yml](mui/material-ui/dependencies/check-if-pr-has-label.png)
![Dependencies ci-check.yml](mui/material-ui/dependencies/ci-check.png)
![Dependencies ci.yml](mui/material-ui/dependencies/ci.png)
![Dependencies codeql.yml](mui/material-ui/dependencies/codeql.png)
![Dependencies maintenance.yml](mui/material-ui/dependencies/maintenance.png)
![Dependencies mark-duplicate.yml](mui/material-ui/dependencies/mark-duplicate.png)
![Dependencies no-response.yml](mui/material-ui/dependencies/no-response.png)
![Dependencies scorecards.yml](mui/material-ui/dependencies/scorecards.png)
![Dependencies support-stackoverflow.yml](mui/material-ui/dependencies/support-stackoverflow.png)

### trekhleb/javascript-algorithms

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.59%      | 2                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.29%      | 1                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name               | Up to date |
|-------------|---------------------------|------------|
| GITHUB      | actions/checkout@v2       | False      |
| GITHUB      | actions/setup-node@v1     | False      |
| PUBLIC      | codecov/codecov-action@v1 | False      |

#### Precedence

![Precedence CI.yml](trekhleb/javascript-algorithms/precedence/CI.png)

#### Dependencies

![Dependencies CI.yml](trekhleb/javascript-algorithms/dependencies/CI.png)

### mantinedev/mantine

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.59%      | 2                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name           | Up to date |
|-------------|-----------------------|------------|
| GITHUB      | actions/checkout@v3   | True       |
| GITHUB      | actions/setup-node@v3 | True       |

#### Precedence

![Precedence pull_request.yml](mantinedev/mantine/precedence/pull_request.png)

#### Dependencies

![Dependencies pull_request.yml](mantinedev/mantine/dependencies/pull_request.png)

### mattermost/mattermost-server

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 1.18%      | 4                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                                                                | Up to date |
|-------------|----------------------------------------------------------------------------|------------|
| GITHUB      | actions/checkout@v3                                                        | True       |
| PUBLIC      | github/codeql-action/init@v2                                               | False      |
| PUBLIC      | github/codeql-action/analyze@v2                                            | False      |
| GITHUB      | actions/checkout@ec3a7ce113134d7a93b817d10a8272cb61118579                  | False      |
| GITHUB      | actions/upload-artifact@82c141cc518b40d92cc801eee768e7aafc9c2fa2           | False      |
| PUBLIC      | ossf/scorecard-action@c1aec4ac820532bab364f02a81873c555a0ba3a1             | False      |
| PUBLIC      | github/codeql-action/upload-sarif@5f532563584d71fdef14ee64d17bafb34f751ce5 | False      |

#### Precedence

![Precedence codeql-analysis.yml](mattermost/mattermost-server/precedence/codeql-analysis.png)
![Precedence scorecards-analysis.yml](mattermost/mattermost-server/precedence/scorecards-analysis.png)

#### Dependencies

![Dependencies codeql-analysis.yml](mattermost/mattermost-server/dependencies/codeql-analysis.png)
![Dependencies scorecards-analysis.yml](mattermost/mattermost-server/dependencies/scorecards-analysis.png)

### pynecone-io/pynecone

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 2.06%      | 7                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.59%      | 2                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name             | Up to date |
|-------------|-------------------------|------------|
| GITHUB      | actions/checkout@v3     | True       |
| GITHUB      | actions/setup-python@v4 | True       |
| GITHUB      | actions/cache@v2        | False      |
| PUBLIC      | snok/install-poetry@v1  | True       |
| GITHUB      | actions/checkout@v3     | True       |
| GITHUB      | actions/setup-node@v3   | True       |
| GITHUB      | actions/setup-python@v4 | True       |
| GITHUB      | actions/cache@v2        | False      |
| PUBLIC      | snok/install-poetry@v1  | True       |

#### Precedence

![Precedence build.yml](pynecone-io/pynecone/precedence/build.png)
![Precedence integration.yml](pynecone-io/pynecone/precedence/integration.png)

#### Dependencies

![Dependencies build.yml](pynecone-io/pynecone/dependencies/build.png)
![Dependencies integration.yml](pynecone-io/pynecone/dependencies/integration.png)

### TheAlgorithms/Python

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 2.06%      | 7                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name             | Up to date |
|-------------|-------------------------|------------|
| GITHUB      | actions/checkout@v3     | True       |
| GITHUB      | actions/setup-python@v4 | True       |
| GITHUB      | actions/cache@v3        | True       |
| GITHUB      | actions/checkout@v1     | False      |
| GITHUB      | actions/setup-python@v4 | True       |
| GITHUB      | actions/checkout@v3     | True       |
| GITHUB      | actions/setup-python@v4 | True       |

#### Precedence

![Precedence build.yml](TheAlgorithms/Python/precedence/build.png)
![Precedence directory_writer.yml](TheAlgorithms/Python/precedence/directory_writer.png)
![Precedence project_euler.yml](TheAlgorithms/Python/precedence/project_euler.png)

#### Dependencies

![Dependencies build.yml](TheAlgorithms/Python/dependencies/build.png)
![Dependencies directory_writer.yml](TheAlgorithms/Python/dependencies/directory_writer.png)
![Dependencies project_euler.yml](TheAlgorithms/Python/dependencies/project_euler.png)

### stefanzweifel/git-auto-commit-action

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                        | Up to date |
|-------------|------------------------------------|------------|
| GITHUB      | actions/checkout@v3                | True       |
| GITHUB      | actions/checkout@v3                | True       |
| PUBLIC      | github/super-linter@v4             | True       |
| PUBLIC      | release-drafter/release-drafter@v5 | True       |
| GITHUB      | actions/checkout@v3                | True       |
| PUBLIC      | Actions-R-Us/actions-tagger@latest | False      |

#### Precedence

![Precedence git-auto-commit.yml](stefanzweifel/git-auto-commit-action/precedence/git-auto-commit.png)
![Precedence linter.yml](stefanzweifel/git-auto-commit-action/precedence/linter.png)
![Precedence release-drafter.yml](stefanzweifel/git-auto-commit-action/precedence/release-drafter.png)
![Precedence tests.yml](stefanzweifel/git-auto-commit-action/precedence/tests.png)
![Precedence versioning.yml](stefanzweifel/git-auto-commit-action/precedence/versioning.png)

#### Dependencies

![Dependencies git-auto-commit.yml](stefanzweifel/git-auto-commit-action/dependencies/git-auto-commit.png)
![Dependencies linter.yml](stefanzweifel/git-auto-commit-action/dependencies/linter.png)
![Dependencies release-drafter.yml](stefanzweifel/git-auto-commit-action/dependencies/release-drafter.png)
![Dependencies tests.yml](stefanzweifel/git-auto-commit-action/dependencies/tests.png)
![Dependencies versioning.yml](stefanzweifel/git-auto-commit-action/dependencies/versioning.png)

### axios/axios

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 2.95%      | 10                |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 2.95%      | 10                |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                                | Up to date |
|-------------|--------------------------------------------|------------|
| GITHUB      | actions/checkout@v3                        | True       |
| GITHUB      | actions/setup-node@v3                      | True       |
| GITHUB      | actions/checkout@v3                        | True       |
| PUBLIC      | github/codeql-action/init@v2               | False      |
| PUBLIC      | github/codeql-action/analyze@v2            | False      |
| GITHUB      | actions/checkout@v3                        | True       |
| GITHUB      | actions/setup-node@v3                      | True       |
| PUBLIC      | martinbeentjes/npm-get-version-action@main | False      |
| PUBLIC      | ffurrer2/extract-release-notes@v1          | True       |
| PUBLIC      | mathiasvr/command-output@v1                | True       |
| PUBLIC      | peter-evans/create-pull-request@v4         | True       |
| GITHUB      | actions/checkout@v3                        | True       |
| GITHUB      | actions/setup-node@v3                      | True       |
| PUBLIC      | martinbeentjes/npm-get-version-action@main | False      |
| PUBLIC      | ffurrer2/extract-release-notes@v1          | True       |
| PUBLIC      | rickstaa/action-create-tag@v1              | True       |
| PUBLIC      | ncipollo/release-action@v1                 | True       |
| GITHUB      | actions/checkout@v2                        | False      |
| GITHUB      | actions/setup-node@v3                      | True       |
| GITHUB      | actions/stale@v7                           | True       |

#### Precedence

![Precedence ci.yml](axios/axios/precedence/ci.png)
![Precedence codeql-analysis.yml](axios/axios/precedence/codeql-analysis.png)
![Precedence pr.yml](axios/axios/precedence/pr.png)
![Precedence publish.yml](axios/axios/precedence/publish.png)
![Precedence release.yml](axios/axios/precedence/release.png)
![Precedence stale.yml](axios/axios/precedence/stale.png)

#### Dependencies

![Dependencies ci.yml](axios/axios/dependencies/ci.png)
![Dependencies codeql-analysis.yml](axios/axios/dependencies/codeql-analysis.png)
![Dependencies pr.yml](axios/axios/dependencies/pr.png)
![Dependencies publish.yml](axios/axios/dependencies/publish.png)
![Dependencies release.yml](axios/axios/dependencies/release.png)
![Dependencies stale.yml](axios/axios/dependencies/stale.png)

### raspberrypi/linux

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                | Up to date |
|-------------|----------------------------|------------|
| GITHUB      | actions/checkout@v3        | True       |
| GITHUB      | actions/checkout@v3        | True       |
| GITHUB      | actions/upload-artifact@v3 | True       |

#### Precedence

![Precedence dtoverlaycheck.yml](raspberrypi/linux/precedence/dtoverlaycheck.png)
![Precedence kernel-build.yml](raspberrypi/linux/precedence/kernel-build.png)

#### Dependencies

![Dependencies dtoverlaycheck.yml](raspberrypi/linux/dependencies/dtoverlaycheck.png)
![Dependencies kernel-build.yml](raspberrypi/linux/dependencies/kernel-build.png)

### kamranahmedse/developer-roadmap

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 1.18%      | 4                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |

#### List of actions

| Action type | Action name                        | Up to date |
|-------------|------------------------------------|------------|
| GITHUB      | actions/checkout@v2                | False      |
| GITHUB      | actions/setup-node@v1              | False      |
| PUBLIC      | pnpm/action-setup@v2.2.2           | False      |
| GITHUB      | actions/checkout@v2                | False      |
| GITHUB      | actions/setup-node@v3              | True       |
| PUBLIC      | pnpm/action-setup@v2.2.2           | False      |
| PUBLIC      | peter-evans/create-pull-request@v4 | True       |

#### Precedence

![Precedence aws-costs.yml](kamranahmedse/developer-roadmap/precedence/aws-costs.png)
![Precedence deploy.yml](kamranahmedse/developer-roadmap/precedence/deploy.png)
![Precedence update-deps.yml](kamranahmedse/developer-roadmap/precedence/update-deps.png)

#### Dependencies

![Dependencies aws-costs.yml](kamranahmedse/developer-roadmap/dependencies/aws-costs.png)
![Dependencies deploy.yml](kamranahmedse/developer-roadmap/dependencies/deploy.png)
![Dependencies update-deps.yml](kamranahmedse/developer-roadmap/dependencies/update-deps.png)
