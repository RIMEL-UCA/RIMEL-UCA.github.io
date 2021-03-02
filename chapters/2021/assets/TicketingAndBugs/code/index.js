const GitHub = require('github-api');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

const csvWriter = createCsvWriter({
    path: 'all_tickets.csv',
    header: [
        {id: 'title', title: 'TITLE'},
        {id: 'issue', title: 'ISSUE_ID'},
        {id: 'logs', title: 'LOGS'},
        {id: 'commits', title: 'COMMITS'},
        {id: 'assignes', title: 'ASSIGNES'},
        {id: 'comments', title: 'COMMENTS'},
        {id: 'open', title: 'OPEN'}
    ]
});

const csvWriterOpen = createCsvWriter({
    path: 'open_tickets.csv',
    header: [
        {id: 'title', title: 'TITLE'},
        {id: 'issue', title: 'ISSUE_ID'},
        {id: 'logs', title: 'LOGS'},
        {id: 'commits', title: 'COMMITS'},
        {id: 'assignes', title: 'ASSIGNES'},
        {id: 'comments', title: 'COMMENTS'},
    ]
});

const csvWriterClosed = createCsvWriter({
    path: 'closed_tickets.csv',
    header: [
        {id: 'title', title: 'TITLE'},
        {id: 'issue', title: 'ISSUE_ID'},
        {id: 'logs', title: 'LOGS'},
        {id: 'commits', title: 'COMMITS'},
        {id: 'assignes', title: 'ASSIGNES'},
        {id: 'comments', title: 'COMMENTS'},
    ]
});

const records = [];
const openRecords = [];
const closedRecords = [];

// basic auth
const gh = new GitHub({
    token: 'YOUR_GITHUB_TOKEN'
});

const user = "angular";
const repo = "angular";
const keyWords = "bug report";

let count = 0;

// partir d'un projet
// mots-clé en entrée
// chercher les tickets correspondants
// extraire des données d'un ticket

const search = gh.search().forIssues({q: "repo:" + user + "/" + repo + " " + keyWords + " in:title,body type:issue"})
    .then(async ({data: issues}) => {
        for (const issue of issues) {
            let commits = 0;
            count += 1;

            console.log("count " + count);

            const title = issue.title.replace(/,/g, ' ');
            const logs = containsLogs(issue.body) ? 1 : 0;

            try {
                // Retriving comments of the issue
                // const comments = await gh.getIssues(user, repo).listIssueComments(issue.number);

                // Retriving events of the issue
                const events = await gh.getIssues(user, repo).listIssueEvents(issue.number);

                for (const event of events.data) {
                    if (event.commit_id) {
                        commits += 1;
                    }
                }

                records.push({
                    title: title, issue: issue.number, logs: logs, commits: commits,
                    assignes: issue.assignees.length, comments: issue.comments, open: issue.state === 'open' ? 1 : 0
                })

                if (issue.state === 'open') {
                    openRecords.push({
                        title: title, issue: issue.number, logs: logs, commits: commits,
                        assignes: issue.assignees.length, comments: issue.comments
                    })
                } else {
                    closedRecords.push({
                        title: title, issue: issue.number, logs: logs, commits: commits,
                        assignes: issue.assignees.length, comments: issue.comments
                    })
                }
            } catch (err) {
                console.error(err);
            }
        }

        csvWriter.writeRecords(records)
            .then(() => {
                console.log('...ALL Done');
            });

        csvWriterOpen.writeRecords(openRecords)
            .then(() => {
                console.log('...OPEN Done');
            });

        csvWriterClosed.writeRecords(closedRecords)
            .then(() => {
                console.log('...CLOSED Done');
            });

    }).catch((error) => {
        console.log('forRepositories error:', error)
    })

//search on the exception or error part if there is a log
function containsLogs(text) {
    if (text == null) return false;

    const separatedOnAt = text.split('Exception or Error');

    if (separatedOnAt.length > 1) {
        const logPart = separatedOnAt[1].split('Your Environment')
        return logPart[0].includes("<code>");
    }
    return false;
}