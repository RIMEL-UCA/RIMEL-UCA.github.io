# my_random_research

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;margin-bottom: 20px;}
.tg td{font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-size:14px;font-weight:bold;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
blockquote { background-color: #cecece !important; padding: 14px 10px 5px 20px !important; border-radius: 5px !important;}
.tgimg { border-radius: 5px!important; max-width: 90%; margin: 15px 5%;}
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
{: .tg}

#### List of actions

| Action type | Action name             | Up to date |
|-------------|-------------------------|------------|
| GITHUB      | actions/checkout@v2     | False      |
| GITHUB      | actions/setup-python@v2 | False      |
| GITHUB      | actions/checkout@v2     | False      |
| GITHUB      | actions/setup-python@v2 | False      |
| GITHUB      | actions/checkout@v2     | False      |
| GITHUB      | actions/setup-python@v2 | False      |
{: .tg}

#### Precedence

![Precedence test_of_push_and_pull.yml](public-apis/public-apis/precedence/test_of_push_and_pull.png)
{: .tgimg}
![Precedence test_of_validate_package.yml](public-apis/public-apis/precedence/test_of_validate_package.png)
{: .tgimg}
![Precedence validate_links.yml](public-apis/public-apis/precedence/validate_links.png)
{: .tgimg}

#### Dependencies

![Dependencies test_of_push_and_pull.yml](public-apis/public-apis/dependencies/test_of_push_and_pull.png)
{: .tgimg}
![Dependencies test_of_validate_package.yml](public-apis/public-apis/dependencies/test_of_validate_package.png)
{: .tgimg}
![Dependencies validate_links.yml](public-apis/public-apis/dependencies/validate_links.png)
{: .tgimg}

### vercel/next.js

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 4.42%      | 15                |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 2.65%      | 9                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence build_test_deploy.yml](vercel/next.js/precedence/build_test_deploy.png)
{: .tgimg}
![Precedence cancel.yml](vercel/next.js/precedence/cancel.png)
{: .tgimg}
![Precedence issue_labeler.yml](vercel/next.js/precedence/issue_labeler.png)
{: .tgimg}
![Precedence issue_lock.yml](vercel/next.js/precedence/issue_lock.png)
{: .tgimg}
![Precedence issue_on_comment.yml](vercel/next.js/precedence/issue_on_comment.png)
{: .tgimg}
![Precedence issue_on_label.yml](vercel/next.js/precedence/issue_on_label.png)
{: .tgimg}
![Precedence issue_stale.yml](vercel/next.js/precedence/issue_stale.png)
{: .tgimg}
![Precedence issue_validator.yml](vercel/next.js/precedence/issue_validator.png)
{: .tgimg}
![Precedence notify_release.yml](vercel/next.js/precedence/notify_release.png)
{: .tgimg}
![Precedence pull_request_stats.yml](vercel/next.js/precedence/pull_request_stats.png)
{: .tgimg}

#### Dependencies

![Dependencies build_test_deploy.yml](vercel/next.js/dependencies/build_test_deploy.png)
{: .tgimg}
![Dependencies cancel.yml](vercel/next.js/dependencies/cancel.png)
{: .tgimg}
![Dependencies issue_labeler.yml](vercel/next.js/dependencies/issue_labeler.png)
{: .tgimg}
![Dependencies issue_lock.yml](vercel/next.js/dependencies/issue_lock.png)
{: .tgimg}
![Dependencies issue_on_comment.yml](vercel/next.js/dependencies/issue_on_comment.png)
{: .tgimg}
![Dependencies issue_on_label.yml](vercel/next.js/dependencies/issue_on_label.png)
{: .tgimg}
![Dependencies issue_stale.yml](vercel/next.js/dependencies/issue_stale.png)
{: .tgimg}
![Dependencies issue_validator.yml](vercel/next.js/dependencies/issue_validator.png)
{: .tgimg}
![Dependencies notify_release.yml](vercel/next.js/dependencies/notify_release.png)
{: .tgimg}
![Dependencies pull_request_stats.yml](vercel/next.js/dependencies/pull_request_stats.png)
{: .tgimg}

### twbs/bootstrap

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 5.01%      | 17                |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 2.95%      | 10                |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence browserstack.yml](twbs/bootstrap/precedence/browserstack.png)
{: .tgimg}
![Precedence bundlewatch.yml](twbs/bootstrap/precedence/bundlewatch.png)
{: .tgimg}
![Precedence calibreapp-image-actions.yml](twbs/bootstrap/precedence/calibreapp-image-actions.png)
{: .tgimg}
![Precedence codeql.yml](twbs/bootstrap/precedence/codeql.png)
{: .tgimg}
![Precedence cspell.yml](twbs/bootstrap/precedence/cspell.png)
{: .tgimg}
![Precedence css.yml](twbs/bootstrap/precedence/css.png)
{: .tgimg}
![Precedence docs.yml](twbs/bootstrap/precedence/docs.png)
{: .tgimg}
![Precedence issue-close-require.yml](twbs/bootstrap/precedence/issue-close-require.png)
{: .tgimg}
![Precedence issue-labeled.yml](twbs/bootstrap/precedence/issue-labeled.png)
{: .tgimg}
![Precedence js.yml](twbs/bootstrap/precedence/js.png)
{: .tgimg}
![Precedence lint.yml](twbs/bootstrap/precedence/lint.png)
{: .tgimg}
![Precedence node-sass.yml](twbs/bootstrap/precedence/node-sass.png)
{: .tgimg}
![Precedence release-notes.yml](twbs/bootstrap/precedence/release-notes.png)
{: .tgimg}

#### Dependencies

![Dependencies browserstack.yml](twbs/bootstrap/dependencies/browserstack.png)
{: .tgimg}
![Dependencies bundlewatch.yml](twbs/bootstrap/dependencies/bundlewatch.png)
{: .tgimg}
![Dependencies calibreapp-image-actions.yml](twbs/bootstrap/dependencies/calibreapp-image-actions.png)
{: .tgimg}
![Dependencies codeql.yml](twbs/bootstrap/dependencies/codeql.png)
{: .tgimg}
![Dependencies cspell.yml](twbs/bootstrap/dependencies/cspell.png)
{: .tgimg}
![Dependencies css.yml](twbs/bootstrap/dependencies/css.png)
{: .tgimg}
![Dependencies docs.yml](twbs/bootstrap/dependencies/docs.png)
{: .tgimg}
![Dependencies issue-close-require.yml](twbs/bootstrap/dependencies/issue-close-require.png)
{: .tgimg}
![Dependencies issue-labeled.yml](twbs/bootstrap/dependencies/issue-labeled.png)
{: .tgimg}
![Dependencies js.yml](twbs/bootstrap/dependencies/js.png)
{: .tgimg}
![Dependencies lint.yml](twbs/bootstrap/dependencies/lint.png)
{: .tgimg}
![Dependencies node-sass.yml](twbs/bootstrap/dependencies/node-sass.png)
{: .tgimg}
![Dependencies release-notes.yml](twbs/bootstrap/dependencies/release-notes.png)
{: .tgimg}

### ytdl-org/youtube-dl

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name             | Up to date |
|-------------|-------------------------|------------|
| GITHUB      | actions/checkout@v2     | False      |
| GITHUB      | actions/setup-python@v2 | False      |
| GITHUB      | actions/setup-java@v1   | False      |
{: .tg}

#### Precedence

![Precedence ci.yml](ytdl-org/youtube-dl/precedence/ci.png)
{: .tgimg}

#### Dependencies

![Dependencies ci.yml](ytdl-org/youtube-dl/dependencies/ci.png)
{: .tgimg}

### vuejs/vue

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.29%      | 1                 |
| TRUSTED     | 0.29%      | 1                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name                  | Up to date |
|-------------|------------------------------|------------|
| GITHUB      | actions/checkout@v2          | False      |
| GITHUB      | actions/setup-node@v2        | False      |
| PUBLIC      | pnpm/action-setup@v2         | True       |
| GITHUB      | actions/checkout@master      | False      |
| TRUSTED     | yyx990803/release-tag@master | False      |
{: .tg}

#### Precedence

![Precedence ci.yml](vuejs/vue/precedence/ci.png)
{: .tgimg}
![Precedence release-tag.yml](vuejs/vue/precedence/release-tag.png)
{: .tgimg}

#### Dependencies

![Dependencies ci.yml](vuejs/vue/dependencies/ci.png)
{: .tgimg}
![Dependencies release-tag.yml](vuejs/vue/dependencies/release-tag.png)
{: .tgimg}

### vuejs/docs

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.0%       | 0                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.29%      | 1                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name                      | Up to date |
|-------------|----------------------------------|------------|
| PUBLIC      | dependabot/fetch-metadata@v1.1.1 | False      |
{: .tg}

#### Precedence

![Precedence automerge.yml](vuejs/docs/precedence/automerge.png)
{: .tgimg}

#### Dependencies

![Dependencies automerge.yml](vuejs/docs/dependencies/automerge.png)
{: .tgimg}

### vuejs/router

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.29%      | 1                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence release-tag.yml](vuejs/router/precedence/release-tag.png)
{: .tgimg}
![Precedence test.yml](vuejs/router/precedence/test.png)
{: .tgimg}

#### Dependencies

![Dependencies release-tag.yml](vuejs/router/dependencies/release-tag.png)
{: .tgimg}
![Dependencies test.yml](vuejs/router/dependencies/test.png)
{: .tgimg}

### microsoft/FluidFramework

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 3.54%      | 12                |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 3.83%      | 13                |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence codeql-analysis.yml](microsoft/FluidFramework/precedence/codeql-analysis.png)
{: .tgimg}
![Precedence linkcheck-reporter.yml](microsoft/FluidFramework/precedence/linkcheck-reporter.png)
{: .tgimg}
![Precedence merge-commits.yml](microsoft/FluidFramework/precedence/merge-commits.png)
{: .tgimg}
![Precedence pr-labeler.yml](microsoft/FluidFramework/precedence/pr-labeler.png)
{: .tgimg}
![Precedence pr-validation.yml](microsoft/FluidFramework/precedence/pr-validation.png)
{: .tgimg}
![Precedence push-queue.yml](microsoft/FluidFramework/precedence/push-queue.png)
{: .tgimg}
![Precedence stale-branches.yml](microsoft/FluidFramework/precedence/stale-branches.png)
{: .tgimg}
![Precedence website-validation.yml](microsoft/FluidFramework/precedence/website-validation.png)
{: .tgimg}

#### Dependencies

![Dependencies codeql-analysis.yml](microsoft/FluidFramework/dependencies/codeql-analysis.png)
{: .tgimg}
![Dependencies linkcheck-reporter.yml](microsoft/FluidFramework/dependencies/linkcheck-reporter.png)
{: .tgimg}
![Dependencies merge-commits.yml](microsoft/FluidFramework/dependencies/merge-commits.png)
{: .tgimg}
![Dependencies pr-labeler.yml](microsoft/FluidFramework/dependencies/pr-labeler.png)
{: .tgimg}
![Dependencies pr-validation.yml](microsoft/FluidFramework/dependencies/pr-validation.png)
{: .tgimg}
![Dependencies push-queue.yml](microsoft/FluidFramework/dependencies/push-queue.png)
{: .tgimg}
![Dependencies stale-branches.yml](microsoft/FluidFramework/dependencies/stale-branches.png)
{: .tgimg}
![Dependencies website-validation.yml](microsoft/FluidFramework/dependencies/website-validation.png)
{: .tgimg}

### microsoft/TypeScript

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 2.65%      | 9                 |
| INTERNAL    | 0.29%      | 1                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence ci.yml](microsoft/TypeScript/precedence/ci.png)
{: .tgimg}
![Precedence codeql.yml](microsoft/TypeScript/precedence/codeql.png)
{: .tgimg}
![Precedence ensure-related-repos-run-crons.yml](microsoft/TypeScript/precedence/ensure-related-repos-run-crons.png)
{: .tgimg}
![Precedence rich-navigation.yml](microsoft/TypeScript/precedence/rich-navigation.png)
{: .tgimg}
![Precedence sync-wiki.yml](microsoft/TypeScript/precedence/sync-wiki.png)
{: .tgimg}
![Precedence update-lkg.yml](microsoft/TypeScript/precedence/update-lkg.png)
{: .tgimg}

#### Dependencies

![Dependencies ci.yml](microsoft/TypeScript/dependencies/ci.png)
{: .tgimg}
![Dependencies codeql.yml](microsoft/TypeScript/dependencies/codeql.png)
{: .tgimg}
![Dependencies ensure-related-repos-run-crons.yml](microsoft/TypeScript/dependencies/ensure-related-repos-run-crons.png)
{: .tgimg}
![Dependencies rich-navigation.yml](microsoft/TypeScript/dependencies/rich-navigation.png)
{: .tgimg}
![Dependencies sync-wiki.yml](microsoft/TypeScript/dependencies/sync-wiki.png)
{: .tgimg}
![Dependencies update-lkg.yml](microsoft/TypeScript/dependencies/update-lkg.png)
{: .tgimg}

### microsoft/winget-cli

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.0%       | 0                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.29%      | 1                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name                           | Up to date |
|-------------|---------------------------------------|------------|
| PUBLIC      | check-spelling/check-spelling@v0.0.21 | True       |
{: .tg}

#### Precedence

![Precedence spelling.yml](microsoft/winget-cli/precedence/spelling.png)
{: .tgimg}
![Precedence spelling2.yml](microsoft/winget-cli/precedence/spelling2.png)
{: .tgimg}
![Precedence spelling3.yml](microsoft/winget-cli/precedence/spelling3.png)
{: .tgimg}

#### Dependencies

![Dependencies spelling.yml](microsoft/winget-cli/dependencies/spelling.png)
{: .tgimg}
![Dependencies spelling2.yml](microsoft/winget-cli/dependencies/spelling2.png)
{: .tgimg}
![Dependencies spelling3.yml](microsoft/winget-cli/dependencies/spelling3.png)
{: .tgimg}

### microsoft/fluentui-react-native

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.29%      | 1                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name                       | Up to date |
|-------------|-----------------------------------|------------|
| GITHUB      | actions/checkout@v2               | False      |
| PUBLIC      | github/codeql-action/init@v1      | False      |
| PUBLIC      | github/codeql-action/autobuild@v1 | False      |
| PUBLIC      | github/codeql-action/analyze@v1   | False      |
{: .tg}

#### Precedence

![Precedence codeql-analysis.yml](microsoft/fluentui-react-native/precedence/codeql-analysis.png)
{: .tgimg}

#### Dependencies

![Dependencies codeql-analysis.yml](microsoft/fluentui-react-native/dependencies/codeql-analysis.png)
{: .tgimg}

### microsoft/azuredatastudio

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 2.06%      | 7                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.59%      | 2                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence ci.yml](microsoft/azuredatastudio/precedence/ci.png)
{: .tgimg}
![Precedence on-label.yml](microsoft/azuredatastudio/precedence/on-label.png)
{: .tgimg}
![Precedence on-pr-open.yml](microsoft/azuredatastudio/precedence/on-pr-open.png)
{: .tgimg}

#### Dependencies

![Dependencies ci.yml](microsoft/azuredatastudio/dependencies/ci.png)
{: .tgimg}
![Dependencies on-label.yml](microsoft/azuredatastudio/dependencies/on-label.png)
{: .tgimg}
![Dependencies on-pr-open.yml](microsoft/azuredatastudio/dependencies/on-pr-open.png)
{: .tgimg}

### microsoft/vscode

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 9.73%      | 33                |
| INTERNAL    | 0.29%      | 1                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}


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
{: .tg}

#### Precedence

![Precedence author-verified.yml](microsoft/vscode/precedence/author-verified.png)
{: .tgimg}
![Precedence bad-tag.yml](microsoft/vscode/precedence/bad-tag.png)
{: .tgimg}
![Precedence basic.yml](microsoft/vscode/precedence/basic.png)
{: .tgimg}
![Precedence ci.yml](microsoft/vscode/precedence/ci.png)
{: .tgimg}
![Precedence deep-classifier-assign-monitor.yml](microsoft/vscode/precedence/deep-classifier-assign-monitor.png)
{: .tgimg}
![Precedence deep-classifier-runner.yml](microsoft/vscode/precedence/deep-classifier-runner.png)
{: .tgimg}
![Precedence deep-classifier-scraper.yml](microsoft/vscode/precedence/deep-classifier-scraper.png)
{: .tgimg}
![Precedence deep-classifier-unassign-monitor.yml](microsoft/vscode/precedence/deep-classifier-unassign-monitor.png)
{: .tgimg}
![Precedence devcontainer-cache.yml](microsoft/vscode/precedence/devcontainer-cache.png)
{: .tgimg}
![Precedence english-please.yml](microsoft/vscode/precedence/english-please.png)
{: .tgimg}
![Precedence feature-request.yml](microsoft/vscode/precedence/feature-request.png)
{: .tgimg}
![Precedence latest-release-monitor.yml](microsoft/vscode/precedence/latest-release-monitor.png)
{: .tgimg}
![Precedence locker.yml](microsoft/vscode/precedence/locker.png)
{: .tgimg}
![Precedence monaco-editor.yml](microsoft/vscode/precedence/monaco-editor.png)
{: .tgimg}
![Precedence needs-more-info-closer.yml](microsoft/vscode/precedence/needs-more-info-closer.png)
{: .tgimg}
![Precedence no-yarn-lock-changes.yml](microsoft/vscode/precedence/no-yarn-lock-changes.png)
{: .tgimg}
![Precedence on-comment.yml](microsoft/vscode/precedence/on-comment.png)
{: .tgimg}
![Precedence on-label.yml](microsoft/vscode/precedence/on-label.png)
{: .tgimg}
![Precedence on-open.yml](microsoft/vscode/precedence/on-open.png)
{: .tgimg}
![Precedence release-pipeline-labeler.yml](microsoft/vscode/precedence/release-pipeline-labeler.png)
{: .tgimg}
![Precedence rich-navigation.yml](microsoft/vscode/precedence/rich-navigation.png)
{: .tgimg}
![Precedence telemetry.yml](microsoft/vscode/precedence/telemetry.png)
{: .tgimg}
![Precedence test-plan-item-validator.yml](microsoft/vscode/precedence/test-plan-item-validator.png)
{: .tgimg}

#### Dependencies

![Dependencies author-verified.yml](microsoft/vscode/dependencies/author-verified.png)
{: .tgimg}
![Dependencies bad-tag.yml](microsoft/vscode/dependencies/bad-tag.png)
{: .tgimg}
![Dependencies basic.yml](microsoft/vscode/dependencies/basic.png)
{: .tgimg}
![Dependencies ci.yml](microsoft/vscode/dependencies/ci.png)
{: .tgimg}
![Dependencies deep-classifier-assign-monitor.yml](microsoft/vscode/dependencies/deep-classifier-assign-monitor.png)
{: .tgimg}
![Dependencies deep-classifier-runner.yml](microsoft/vscode/dependencies/deep-classifier-runner.png)
{: .tgimg}
![Dependencies deep-classifier-scraper.yml](microsoft/vscode/dependencies/deep-classifier-scraper.png)
{: .tgimg}
![Dependencies deep-classifier-unassign-monitor.yml](microsoft/vscode/dependencies/deep-classifier-unassign-monitor.png)
{: .tgimg}
![Dependencies devcontainer-cache.yml](microsoft/vscode/dependencies/devcontainer-cache.png)
{: .tgimg}
![Dependencies english-please.yml](microsoft/vscode/dependencies/english-please.png)
{: .tgimg}
![Dependencies feature-request.yml](microsoft/vscode/dependencies/feature-request.png)
{: .tgimg}
![Dependencies latest-release-monitor.yml](microsoft/vscode/dependencies/latest-release-monitor.png)
{: .tgimg}
![Dependencies locker.yml](microsoft/vscode/dependencies/locker.png)
{: .tgimg}
![Dependencies monaco-editor.yml](microsoft/vscode/dependencies/monaco-editor.png)
{: .tgimg}
![Dependencies needs-more-info-closer.yml](microsoft/vscode/dependencies/needs-more-info-closer.png)
{: .tgimg}
![Dependencies no-yarn-lock-changes.yml](microsoft/vscode/dependencies/no-yarn-lock-changes.png)
{: .tgimg}
![Dependencies on-comment.yml](microsoft/vscode/dependencies/on-comment.png)
{: .tgimg}
![Dependencies on-label.yml](microsoft/vscode/dependencies/on-label.png)
{: .tgimg}
![Dependencies on-open.yml](microsoft/vscode/dependencies/on-open.png)
{: .tgimg}
![Dependencies release-pipeline-labeler.yml](microsoft/vscode/dependencies/release-pipeline-labeler.png)
{: .tgimg}
![Dependencies rich-navigation.yml](microsoft/vscode/dependencies/rich-navigation.png)
{: .tgimg}
![Dependencies telemetry.yml](microsoft/vscode/dependencies/telemetry.png)
{: .tgimg}
![Dependencies test-plan-item-validator.yml](microsoft/vscode/dependencies/test-plan-item-validator.png)
{: .tgimg}

### collet/cucumber-demo

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.59%      | 2                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name           | Up to date |
|-------------|-----------------------|------------|
| GITHUB      | actions/checkout@v2   | False      |
| GITHUB      | actions/setup-java@v1 | False      |
{: .tg}

#### Precedence

![Precedence maven.yml](collet/cucumber-demo/precedence/maven.png)
{: .tgimg}

#### Dependencies

![Dependencies maven.yml](collet/cucumber-demo/dependencies/maven.png)
{: .tgimg}

### mathiascouste/qgl-template

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.59%      | 2                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name           | Up to date |
|-------------|-----------------------|------------|
| GITHUB      | actions/checkout@v2   | False      |
| GITHUB      | actions/setup-java@v1 | False      |
{: .tg}

#### Precedence

![Precedence pr-build.yml](mathiascouste/qgl-template/precedence/pr-build.png)
{: .tgimg}

#### Dependencies

![Dependencies pr-build.yml](mathiascouste/qgl-template/dependencies/pr-build.png)
{: .tgimg}

### vitest-dev/vitest

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 1.47%      | 5                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence bench.yml](vitest-dev/vitest/precedence/bench.png)
{: .tgimg}
![Precedence ci.yml](vitest-dev/vitest/precedence/ci.png)
{: .tgimg}
![Precedence issue-close-require.yml](vitest-dev/vitest/precedence/issue-close-require.png)
{: .tgimg}
![Precedence issue-labeled.yml](vitest-dev/vitest/precedence/issue-labeled.png)
{: .tgimg}
![Precedence release.yml](vitest-dev/vitest/precedence/release.png)
{: .tgimg}

#### Dependencies

![Dependencies bench.yml](vitest-dev/vitest/dependencies/bench.png)
{: .tgimg}
![Dependencies ci.yml](vitest-dev/vitest/dependencies/ci.png)
{: .tgimg}
![Dependencies issue-close-require.yml](vitest-dev/vitest/dependencies/issue-close-require.png)
{: .tgimg}
![Dependencies issue-labeled.yml](vitest-dev/vitest/dependencies/issue-labeled.png)
{: .tgimg}
![Dependencies release.yml](vitest-dev/vitest/dependencies/release.png)
{: .tgimg}

### i18next/next-i18next

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.59%      | 2                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name           | Up to date |
|-------------|-----------------------|------------|
| GITHUB      | actions/checkout@v3   | True       |
| GITHUB      | actions/setup-node@v3 | True       |
{: .tg}

#### Precedence

![Precedence ci.yml](i18next/next-i18next/precedence/ci.png)
{: .tgimg}

#### Dependencies

![Dependencies ci.yml](i18next/next-i18next/dependencies/ci.png)
{: .tgimg}

### jwasham/coding-interview-university

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.29%      | 1                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 1.18%      | 4                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name                           | Up to date |
|-------------|---------------------------------------|------------|
| GITHUB      | actions/checkout@v3                   | True       |
| PUBLIC      | lycheeverse/lychee-action@v1.4.1      | False      |
| PUBLIC      | micalevisk/last-issue-action@v1.2     | False      |
| PUBLIC      | peter-evans/create-issue-from-file@v4 | True       |
| PUBLIC      | peter-evans/close-issue@v2            | True       |
{: .tg}

#### Precedence

![Precedence links_checker.yml](jwasham/coding-interview-university/precedence/links_checker.png)
{: .tgimg}

#### Dependencies

![Dependencies links_checker.yml](jwasham/coding-interview-university/dependencies/links_checker.png)
{: .tgimg}

### EbookFoundation/free-programming-books

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 1.47%      | 5                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 1.47%      | 5                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence check-urls.yml](EbookFoundation/free-programming-books/precedence/check-urls.png)
{: .tgimg}
![Precedence detect-conflicting-prs.yml](EbookFoundation/free-programming-books/precedence/detect-conflicting-prs.png)
{: .tgimg}
![Precedence fpb-lint.yml](EbookFoundation/free-programming-books/precedence/fpb-lint.png)
{: .tgimg}
![Precedence issues-pinner.yml](EbookFoundation/free-programming-books/precedence/issues-pinner.png)
{: .tgimg}
![Precedence stale.yml](EbookFoundation/free-programming-books/precedence/stale.png)
{: .tgimg}

#### Dependencies

![Dependencies check-urls.yml](EbookFoundation/free-programming-books/dependencies/check-urls.png)
{: .tgimg}
![Dependencies detect-conflicting-prs.yml](EbookFoundation/free-programming-books/dependencies/detect-conflicting-prs.png)
{: .tgimg}
![Dependencies fpb-lint.yml](EbookFoundation/free-programming-books/dependencies/fpb-lint.png)
{: .tgimg}
![Dependencies issues-pinner.yml](EbookFoundation/free-programming-books/dependencies/issues-pinner.png)
{: .tgimg}
![Dependencies stale.yml](EbookFoundation/free-programming-books/dependencies/stale.png)
{: .tgimg}

### flutter/flutter

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 1.18%      | 4                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence coverage.yml](flutter/flutter/precedence/coverage.png)
{: .tgimg}
![Precedence mirror.yml](flutter/flutter/precedence/mirror.png)
{: .tgimg}
![Precedence scorecards-analysis.yml](flutter/flutter/precedence/scorecards-analysis.png)
{: .tgimg}

#### Dependencies

![Dependencies coverage.yml](flutter/flutter/dependencies/coverage.png)
{: .tgimg}
![Dependencies mirror.yml](flutter/flutter/dependencies/mirror.png)
{: .tgimg}
![Dependencies scorecards-analysis.yml](flutter/flutter/dependencies/scorecards-analysis.png)
{: .tgimg}

### mobileandyou/api

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name                  | Up to date |
|-------------|------------------------------|------------|
| GITHUB      | actions/checkout@v2          | False      |
| GITHUB      | actions/setup-java@v1        | False      |
| PUBLIC      | appleboy/scp-action@master   | False      |
| PUBLIC      | appleboy/ssh-action@master   | False      |
| GITHUB      | actions/checkout@v3          | True       |
| PUBLIC      | JetBrains/qodana-action@main | False      |
{: .tg}

#### Precedence

![Precedence api.yml](mobileandyou/api/precedence/api.png)
{: .tgimg}
![Precedence qodana.yml](mobileandyou/api/precedence/qodana.png)
{: .tgimg}

#### Dependencies

![Dependencies api.yml](mobileandyou/api/dependencies/api.png)
{: .tgimg}
![Dependencies qodana.yml](mobileandyou/api/dependencies/qodana.png)
{: .tgimg}

### facebook/react

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 1.77%      | 6                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.29%      | 1                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence commit_artifacts.yml](facebook/react/precedence/commit_artifacts.png)
{: .tgimg}
![Precedence devtools_check_repro.yml](facebook/react/precedence/devtools_check_repro.png)
{: .tgimg}

#### Dependencies

![Dependencies commit_artifacts.yml](facebook/react/dependencies/commit_artifacts.png)
{: .tgimg}
![Dependencies devtools_check_repro.yml](facebook/react/dependencies/devtools_check_repro.png)
{: .tgimg}

### freeCodeCamp/freeCodeCamp

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 8.85%      | 30                |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 5.6%       | 19                |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence codeql-analysis.yml](freeCodeCamp/freeCodeCamp/precedence/codeql-analysis.png)
{: .tgimg}
![Precedence codesee-diagram.yml](freeCodeCamp/freeCodeCamp/precedence/codesee-diagram.png)
{: .tgimg}
![Precedence crowdin-download.client-ui.yml](freeCodeCamp/freeCodeCamp/precedence/crowdin-download.client-ui.png)
{: .tgimg}
![Precedence crowdin-download.curriculum.yml](freeCodeCamp/freeCodeCamp/precedence/crowdin-download.curriculum.png)
{: .tgimg}
![Precedence crowdin-download.docs.yml](freeCodeCamp/freeCodeCamp/precedence/crowdin-download.docs.png)
{: .tgimg}
![Precedence crowdin-upload.client-ui.yml](freeCodeCamp/freeCodeCamp/precedence/crowdin-upload.client-ui.png)
{: .tgimg}
![Precedence crowdin-upload.curriculum.yml](freeCodeCamp/freeCodeCamp/precedence/crowdin-upload.curriculum.png)
{: .tgimg}
![Precedence crowdin-upload.docs.yml](freeCodeCamp/freeCodeCamp/precedence/crowdin-upload.docs.png)
{: .tgimg}
![Precedence e2e-mobile.yml](freeCodeCamp/freeCodeCamp/precedence/e2e-mobile.png)
{: .tgimg}
![Precedence e2e-third-party.yml](freeCodeCamp/freeCodeCamp/precedence/e2e-third-party.png)
{: .tgimg}
![Precedence e2e-web.yml](freeCodeCamp/freeCodeCamp/precedence/e2e-web.png)
{: .tgimg}
![Precedence github-autoclose.yml](freeCodeCamp/freeCodeCamp/precedence/github-autoclose.png)
{: .tgimg}
![Precedence github-no-i18n-via-prs.yml](freeCodeCamp/freeCodeCamp/precedence/github-no-i18n-via-prs.png)
{: .tgimg}
![Precedence github-spam.yml](freeCodeCamp/freeCodeCamp/precedence/github-spam.png)
{: .tgimg}
![Precedence i18n-validate-builds.yml](freeCodeCamp/freeCodeCamp/precedence/i18n-validate-builds.png)
{: .tgimg}
![Precedence i18n-validate-prs.yml](freeCodeCamp/freeCodeCamp/precedence/i18n-validate-prs.png)
{: .tgimg}
![Precedence node.js-find-unused.yml](freeCodeCamp/freeCodeCamp/precedence/node.js-find-unused.png)
{: .tgimg}
![Precedence node.js-tests-upcoming.yml](freeCodeCamp/freeCodeCamp/precedence/node.js-tests-upcoming.png)
{: .tgimg}
![Precedence node.js-tests.yml](freeCodeCamp/freeCodeCamp/precedence/node.js-tests.png)
{: .tgimg}

#### Dependencies

![Dependencies codeql-analysis.yml](freeCodeCamp/freeCodeCamp/dependencies/codeql-analysis.png)
{: .tgimg}
![Dependencies codesee-diagram.yml](freeCodeCamp/freeCodeCamp/dependencies/codesee-diagram.png)
{: .tgimg}
![Dependencies crowdin-download.client-ui.yml](freeCodeCamp/freeCodeCamp/dependencies/crowdin-download.client-ui.png)
{: .tgimg}
![Dependencies crowdin-download.curriculum.yml](freeCodeCamp/freeCodeCamp/dependencies/crowdin-download.curriculum.png)
{: .tgimg}
![Dependencies crowdin-download.docs.yml](freeCodeCamp/freeCodeCamp/dependencies/crowdin-download.docs.png)
{: .tgimg}
![Dependencies crowdin-upload.client-ui.yml](freeCodeCamp/freeCodeCamp/dependencies/crowdin-upload.client-ui.png)
{: .tgimg}
![Dependencies crowdin-upload.curriculum.yml](freeCodeCamp/freeCodeCamp/dependencies/crowdin-upload.curriculum.png)
{: .tgimg}
![Dependencies crowdin-upload.docs.yml](freeCodeCamp/freeCodeCamp/dependencies/crowdin-upload.docs.png)
{: .tgimg}
![Dependencies e2e-mobile.yml](freeCodeCamp/freeCodeCamp/dependencies/e2e-mobile.png)
{: .tgimg}
![Dependencies e2e-third-party.yml](freeCodeCamp/freeCodeCamp/dependencies/e2e-third-party.png)
{: .tgimg}
![Dependencies e2e-web.yml](freeCodeCamp/freeCodeCamp/dependencies/e2e-web.png)
{: .tgimg}
![Dependencies github-autoclose.yml](freeCodeCamp/freeCodeCamp/dependencies/github-autoclose.png)
{: .tgimg}
![Dependencies github-no-i18n-via-prs.yml](freeCodeCamp/freeCodeCamp/dependencies/github-no-i18n-via-prs.png)
{: .tgimg}
![Dependencies github-spam.yml](freeCodeCamp/freeCodeCamp/dependencies/github-spam.png)
{: .tgimg}
![Dependencies i18n-validate-builds.yml](freeCodeCamp/freeCodeCamp/dependencies/i18n-validate-builds.png)
{: .tgimg}
![Dependencies i18n-validate-prs.yml](freeCodeCamp/freeCodeCamp/dependencies/i18n-validate-prs.png)
{: .tgimg}
![Dependencies node.js-find-unused.yml](freeCodeCamp/freeCodeCamp/dependencies/node.js-find-unused.png)
{: .tgimg}
![Dependencies node.js-tests-upcoming.yml](freeCodeCamp/freeCodeCamp/dependencies/node.js-tests-upcoming.png)
{: .tgimg}
![Dependencies node.js-tests.yml](freeCodeCamp/freeCodeCamp/dependencies/node.js-tests.png)
{: .tgimg}

### d3/d3

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.59%      | 2                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name           | Up to date |
|-------------|-----------------------|------------|
| GITHUB      | actions/checkout@v2   | False      |
| GITHUB      | actions/setup-node@v1 | False      |
{: .tg}

#### Precedence

![Precedence node.js.yml](d3/d3/precedence/node.js.png)
{: .tgimg}

#### Dependencies

![Dependencies node.js.yml](d3/d3/dependencies/node.js.png)
{: .tgimg}

### mui/material-ui

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 1.18%      | 4                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 2.65%      | 9                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence check-if-pr-has-label.yml](mui/material-ui/precedence/check-if-pr-has-label.png)
{: .tgimg}
![Precedence ci-check.yml](mui/material-ui/precedence/ci-check.png)
{: .tgimg}
![Precedence ci.yml](mui/material-ui/precedence/ci.png)
{: .tgimg}
![Precedence codeql.yml](mui/material-ui/precedence/codeql.png)
{: .tgimg}
![Precedence maintenance.yml](mui/material-ui/precedence/maintenance.png)
{: .tgimg}
![Precedence mark-duplicate.yml](mui/material-ui/precedence/mark-duplicate.png)
{: .tgimg}
![Precedence no-response.yml](mui/material-ui/precedence/no-response.png)
{: .tgimg}
![Precedence scorecards.yml](mui/material-ui/precedence/scorecards.png)
{: .tgimg}
![Precedence support-stackoverflow.yml](mui/material-ui/precedence/support-stackoverflow.png)
{: .tgimg}

#### Dependencies

![Dependencies check-if-pr-has-label.yml](mui/material-ui/dependencies/check-if-pr-has-label.png)
{: .tgimg}
![Dependencies ci-check.yml](mui/material-ui/dependencies/ci-check.png)
{: .tgimg}
![Dependencies ci.yml](mui/material-ui/dependencies/ci.png)
{: .tgimg}
![Dependencies codeql.yml](mui/material-ui/dependencies/codeql.png)
{: .tgimg}
![Dependencies maintenance.yml](mui/material-ui/dependencies/maintenance.png)
{: .tgimg}
![Dependencies mark-duplicate.yml](mui/material-ui/dependencies/mark-duplicate.png)
{: .tgimg}
![Dependencies no-response.yml](mui/material-ui/dependencies/no-response.png)
{: .tgimg}
![Dependencies scorecards.yml](mui/material-ui/dependencies/scorecards.png)
{: .tgimg}
![Dependencies support-stackoverflow.yml](mui/material-ui/dependencies/support-stackoverflow.png)
{: .tgimg}

### trekhleb/javascript-algorithms

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.59%      | 2                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.29%      | 1                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name               | Up to date |
|-------------|---------------------------|------------|
| GITHUB      | actions/checkout@v2       | False      |
| GITHUB      | actions/setup-node@v1     | False      |
| PUBLIC      | codecov/codecov-action@v1 | False      |
{: .tg}

#### Precedence

![Precedence CI.yml](trekhleb/javascript-algorithms/precedence/CI.png)
{: .tgimg}

#### Dependencies

![Dependencies CI.yml](trekhleb/javascript-algorithms/dependencies/CI.png)
{: .tgimg}

### mantinedev/mantine

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.59%      | 2                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name           | Up to date |
|-------------|-----------------------|------------|
| GITHUB      | actions/checkout@v3   | True       |
| GITHUB      | actions/setup-node@v3 | True       |
{: .tg}

#### Precedence

![Precedence pull_request.yml](mantinedev/mantine/precedence/pull_request.png)
{: .tgimg}

#### Dependencies

![Dependencies pull_request.yml](mantinedev/mantine/dependencies/pull_request.png)
{: .tgimg}

### mattermost/mattermost-server

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 1.18%      | 4                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence codeql-analysis.yml](mattermost/mattermost-server/precedence/codeql-analysis.png)
{: .tgimg}
![Precedence scorecards-analysis.yml](mattermost/mattermost-server/precedence/scorecards-analysis.png)
{: .tgimg}

#### Dependencies

![Dependencies codeql-analysis.yml](mattermost/mattermost-server/dependencies/codeql-analysis.png)
{: .tgimg}
![Dependencies scorecards-analysis.yml](mattermost/mattermost-server/dependencies/scorecards-analysis.png)
{: .tgimg}

### pynecone-io/pynecone

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 2.06%      | 7                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.59%      | 2                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence build.yml](pynecone-io/pynecone/precedence/build.png)
{: .tgimg}
![Precedence integration.yml](pynecone-io/pynecone/precedence/integration.png)
{: .tgimg}

#### Dependencies

![Dependencies build.yml](pynecone-io/pynecone/dependencies/build.png)
{: .tgimg}
![Dependencies integration.yml](pynecone-io/pynecone/dependencies/integration.png)
{: .tgimg}

### TheAlgorithms/Python

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 2.06%      | 7                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence build.yml](TheAlgorithms/Python/precedence/build.png)
{: .tgimg}
![Precedence directory_writer.yml](TheAlgorithms/Python/precedence/directory_writer.png)
{: .tgimg}
![Precedence project_euler.yml](TheAlgorithms/Python/precedence/project_euler.png)
{: .tgimg}

#### Dependencies

![Dependencies build.yml](TheAlgorithms/Python/dependencies/build.png)
{: .tgimg}
![Dependencies directory_writer.yml](TheAlgorithms/Python/dependencies/directory_writer.png)
{: .tgimg}
![Dependencies project_euler.yml](TheAlgorithms/Python/dependencies/project_euler.png)
{: .tgimg}

### stefanzweifel/git-auto-commit-action

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name                        | Up to date |
|-------------|------------------------------------|------------|
| GITHUB      | actions/checkout@v3                | True       |
| GITHUB      | actions/checkout@v3                | True       |
| PUBLIC      | github/super-linter@v4             | True       |
| PUBLIC      | release-drafter/release-drafter@v5 | True       |
| GITHUB      | actions/checkout@v3                | True       |
| PUBLIC      | Actions-R-Us/actions-tagger@latest | False      |
{: .tg}

#### Precedence

![Precedence git-auto-commit.yml](stefanzweifel/git-auto-commit-action/precedence/git-auto-commit.png)
{: .tgimg}
![Precedence linter.yml](stefanzweifel/git-auto-commit-action/precedence/linter.png)
{: .tgimg}
![Precedence release-drafter.yml](stefanzweifel/git-auto-commit-action/precedence/release-drafter.png)
{: .tgimg}
![Precedence tests.yml](stefanzweifel/git-auto-commit-action/precedence/tests.png)
{: .tgimg}
![Precedence versioning.yml](stefanzweifel/git-auto-commit-action/precedence/versioning.png)
{: .tgimg}

#### Dependencies

![Dependencies git-auto-commit.yml](stefanzweifel/git-auto-commit-action/dependencies/git-auto-commit.png)
{: .tgimg}
![Dependencies linter.yml](stefanzweifel/git-auto-commit-action/dependencies/linter.png)
{: .tgimg}
![Dependencies release-drafter.yml](stefanzweifel/git-auto-commit-action/dependencies/release-drafter.png)
{: .tgimg}
![Dependencies tests.yml](stefanzweifel/git-auto-commit-action/dependencies/tests.png)
{: .tgimg}
![Dependencies versioning.yml](stefanzweifel/git-auto-commit-action/dependencies/versioning.png)
{: .tgimg}

### axios/axios

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 2.95%      | 10                |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 2.95%      | 10                |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence ci.yml](axios/axios/precedence/ci.png)
{: .tgimg}
![Precedence codeql-analysis.yml](axios/axios/precedence/codeql-analysis.png)
{: .tgimg}
![Precedence pr.yml](axios/axios/precedence/pr.png)
{: .tgimg}
![Precedence publish.yml](axios/axios/precedence/publish.png)
{: .tgimg}
![Precedence release.yml](axios/axios/precedence/release.png)
{: .tgimg}
![Precedence stale.yml](axios/axios/precedence/stale.png)
{: .tgimg}

#### Dependencies

![Dependencies ci.yml](axios/axios/dependencies/ci.png)
{: .tgimg}
![Dependencies codeql-analysis.yml](axios/axios/dependencies/codeql-analysis.png)
{: .tgimg}
![Dependencies pr.yml](axios/axios/dependencies/pr.png)
{: .tgimg}
![Dependencies publish.yml](axios/axios/dependencies/publish.png)
{: .tgimg}
![Dependencies release.yml](axios/axios/dependencies/release.png)
{: .tgimg}
![Dependencies stale.yml](axios/axios/dependencies/stale.png)
{: .tgimg}

### raspberrypi/linux

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 0.88%      | 3                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.0%       | 0                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

#### List of actions

| Action type | Action name                | Up to date |
|-------------|----------------------------|------------|
| GITHUB      | actions/checkout@v3        | True       |
| GITHUB      | actions/checkout@v3        | True       |
| GITHUB      | actions/upload-artifact@v3 | True       |
{: .tg}

#### Precedence

![Precedence dtoverlaycheck.yml](raspberrypi/linux/precedence/dtoverlaycheck.png)
{: .tgimg}
![Precedence kernel-build.yml](raspberrypi/linux/precedence/kernel-build.png)
{: .tgimg}

#### Dependencies

![Dependencies dtoverlaycheck.yml](raspberrypi/linux/dependencies/dtoverlaycheck.png)
{: .tgimg}
![Dependencies kernel-build.yml](raspberrypi/linux/dependencies/kernel-build.png)
{: .tgimg}

### kamranahmedse/developer-roadmap

#### Repartition of actions types

| Action type | Percentage | Number of actions |
|-------------|------------|-------------------|
| GITHUB      | 1.18%      | 4                 |
| INTERNAL    | 0.0%       | 0                 |
| PUBLIC      | 0.88%      | 3                 |
| TRUSTED     | 0.0%       | 0                 |
| FORKED      | 0.0%       | 0                 |
{: .tg}

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
{: .tg}

#### Precedence

![Precedence aws-costs.yml](kamranahmedse/developer-roadmap/precedence/aws-costs.png)
{: .tgimg}
![Precedence deploy.yml](kamranahmedse/developer-roadmap/precedence/deploy.png)
{: .tgimg}
![Precedence update-deps.yml](kamranahmedse/developer-roadmap/precedence/update-deps.png)
{: .tgimg}

#### Dependencies

![Dependencies aws-costs.yml](kamranahmedse/developer-roadmap/dependencies/aws-costs.png)
{: .tgimg}
![Dependencies deploy.yml](kamranahmedse/developer-roadmap/dependencies/deploy.png)
{: .tgimg}
![Dependencies update-deps.yml](kamranahmedse/developer-roadmap/dependencies/update-deps.png)
{: .tgimg}
