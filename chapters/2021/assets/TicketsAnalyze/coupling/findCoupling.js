const express  = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const app =  express();
const axios = require("axios");
const jsdom = require("jsdom");
const { JSDOM } = jsdom;


app.use(bodyParser.json());
app.use(bodyParser.urlencoded({
    extended: true
}));

let directoryToComponents = [];
let numberOfTicketsWithGitHubLink = 0;
let componentsFiles;
let tickets = undefined;
let maxResults = 10000
const url = "https://jira.mongodb.org/rest/api/2/search?jql=project=SERVER&maxResults="+maxResults;

function getTickets() {
    let promises = [];
    const fs = require('fs');
    fs.readFile('./1000.json', 'utf8', (err, data) => {
        if (err) {
            console.log(`Error reading file from disk: ${err}`);
        } else {
            componentsFiles = JSON.parse(data);
        }

    });
    for(let i=0; i<51848; i+=1000){
        return axios.get(url, { headers: { Accept: "application/json" } })
        .then(res => {
            tickets = res.data;
            let numberOfTickets = 0;
            tickets.issues.forEach(ticket => { 
                let description = ticket.fields.description;
                let components = ticket.fields.components;
                if(components.length == 2) {
                    numberOfTickets += 1;
                    if(description!== null && description.includes("github.com")){
                        numberOfTicketsWithGitHubLink += 1;
                        let links = description.split("github.com").slice(1).map(e => "https://www.github.com" + e.split(" ")[0].split("]")[0].split("#")[0].split("\n")[0].trim());
                        links.forEach(link => {
                            try{
                                if(link !== ""){
                                    promises.push(findInclude(link, components[0].name, components[1].name, ticket.id));
                                }       
                            } catch(error){
                                console.log('error here');
                            }
                        })
                    }
                }
            });
            Promise.all(promises).then(() => {
                directoryToComponents.push({numberOfTicketsWith2Components: numberOfTickets});
                directoryToComponents.push({numberOfTicketsAnalyzed: numberOfTicketsWithGitHubLink});
                fs.writeFile("result-"+maxResults.toString()+".json", JSON.stringify(directoryToComponents, null, 1), 'utf8', function (err) {
                    if (err) {
                        console.log("An error occured while writing JSON Object to File.");
                        return console.log(err);
                    }
                    console.log("JSON file has been saved.");
                });
            });
        });
    }
    
}

getTickets();



function addToJSON(componentName1, componentName2, url, ticketId){
    //console.log('waiteddd22222');
    if(directoryToComponents.find(element => element.ticketId === ticketId) === undefined){
        directoryToComponents.push({
            ticketId: ticketId,
            links: [
                {
                    url: url,
                    fileComponent: componentName1,
                    includeComponent: componentName2,
                    numberOfIncludesFound: 1
                }
            ]
        });
    } else {
        let links = directoryToComponents[directoryToComponents.findIndex(element => element.ticketId === ticketId)].links;
        let bool = false;
        links.forEach(link =>{
            if(link.url === url){
                directoryToComponents[directoryToComponents.findIndex(element => element.ticketId === ticketId)].links[links.findIndex(e => e.url === url)].numberOfIncludesFound += 1;
                bool = true;
            }
        });
        if(! bool){
            directoryToComponents[directoryToComponents.findIndex(element => element.ticketId === ticketId)].links.push({
                links: [
                    {
                        url: url,
                        fileComponent: componentName1,
                        includeComponent: componentName2,
                        numberOfIncludesFound: 1
                    }
                ]
            })
        }
    }
}

async function findInclude(url, componentName1, componentName2, ticketId){
    await axios.get(url, { headers: { Accept: "application/json" } })
    .then(res => {
        if(url.includes('mongo')){
            //console.log('components: ', componentName1, componentName2);
            const html = new JSDOM(res.data).window.document;
            let filteredHtml = html.documentElement.getElementsByClassName('blob-code blob-code-inner js-file-line');
            if(url.includes('commit')){
                for(let i =0; i<filteredHtml.length; i++){
                    if(filteredHtml[i].textContent.includes('src')){
                        //console.log('FILENAME: ', filteredHtml[i].textContent.split('src/')[1].split(' ')[0].split('\n')[0]);
                        getComponent(componentName1, componentName2, filteredHtml[i].textContent.split('src/')[1].split(' ')[0].split('\n')[0], filteredHtml);
                    }
                }
            }
            else if(url.includes('/src') || url.includes('/jstests')){//Good url
                if(url.includes('src/')){
                    getComponent(componentName1, componentName2, 'src/'+url.split('src/')[1], filteredHtml, url, ticketId);
                }
                if(url.includes('/jstests')){
                     getComponent(componentName1, componentName2, 'jstests/'+url.split('jstests/')[1], filteredHtml, url, ticketId);
                }
            }
            else {
                //console.log('bad link');
            }
        }
    })
    .catch(err => {
        //console.error('error: '+url);
    });
}

function getComponent(componentName1, componentName2, fileName, filteredHtml, url, ticketId){
    let splittedFileName = fileName.split('/');
    splittedFileName.splice(-1); //Remove the last element (filename without path)
    let pathName = ""; //Path to file without file name
    splittedFileName.forEach(directory => {
        pathName+= directory + '/';
    });
    //console.log('FILENAME', pathName);
    let component1NumberOfIncludesFound = 0;
    let component2NumberOfIncludesFound = 0;
    componentsFiles.forEach(component => {
        if(component.name === componentName1){
            component.paths.forEach(path => {
                if(path.path.includes(pathName)){
                    //console.log('Found:',path.path, fileName);
                    component1NumberOfIncludesFound += 1;
                }
            });
        } else if(component.name === componentName2){
            component.paths.forEach(path => {
                if(path.path.includes(pathName)){
                    //console.log('Found:',path.path, fileName);
                    component2NumberOfIncludesFound +=1;
                }
            });
        }
    });
    if(component1NumberOfIncludesFound !== 0 || component2NumberOfIncludesFound !==0){
        if(component1NumberOfIncludesFound >= component2NumberOfIncludesFound){
            searchCoupling(componentName1, filteredHtml, componentName2, url,  ticketId);
        } else {
            searchCoupling(componentName2, filteredHtml, componentName1,url, ticketId);
        }
    }
}

function getComponentFromFilename(fileName) {
    let splittedFileName = fileName.split('/');
    splittedFileName.splice(-1); //Remove the last element (filename without path)
    let pathName = ""; //Path to file without file name
    splittedFileName.forEach(directory => {
        pathName+= directory + '/';
    });
    let coupling = {};
    componentsFiles.forEach(component => {
        component.paths.forEach(path => {
            if(path.path.includes(pathName)){
                if(coupling[component.name] === undefined){
                    //console.log('first found:', component.name);
                    coupling[component.name] = 1;
                } else {
                    coupling[component.name] += 1;
                }
            }
        })
    });
    let maxValue = -1;
    let componentName;
    for (let [key, value] of Object.entries(coupling)) {
        if(value > maxValue){
            maxValue = value;
            componentName = key;
        }
    }
    return componentName;
}

function searchCoupling(componentName, filteredHtml, componentName2, url, ticketId){
    for(let i =0; i<filteredHtml.length; i++){
        if(filteredHtml[i].textContent.startsWith('#include') && filteredHtml[i].textContent.includes('mongo/')){
            let coupledComponentName = getComponentFromFilename(filteredHtml[i].textContent.split('mongo/')[1].split('"')[0]);
            if(coupledComponentName !== undefined && coupledComponentName !== null){
                if(coupledComponentName == componentName2){
                    addToJSON(componentName, componentName2, url, ticketId);
                }
            }
        }
    }
}