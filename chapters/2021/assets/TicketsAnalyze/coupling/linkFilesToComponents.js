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
let tickets = undefined;
let maxResults = 1000;
const url = "https://jira.mongodb.org/rest/api/2/search?jql=project=SERVER&maxResults="+maxResults.toString();

function getTickets() {
    let promises = [];
    return axios.get(url, { headers: { Accept: "application/json" } })
    .then(res => {
        tickets = res.data;
        tickets.issues.forEach(ticket => { 
            let description = ticket.fields.description;
            let components = ticket.fields.components;
            if(components.length == 1) {
                if(description!== null && description.includes("github.com")){
                    let links = description.split("github.com").slice(1).map(e => "https://www.github.com" + e.split(" ")[0].split("]")[0].split("#")[0].split("\n")[0].trim());
                    links.forEach(link => {
                        try{
                            if(link !== ""){
                                promises.push(findLink(link, components[0].name));
                            }       
                        } catch(error){
                            //console.log(error);
                        }
                    })
                }
            }
        });
        Promise.all(promises).then(() => {
            fs.writeFile(maxResults.toString()+".json", JSON.stringify(directoryToComponents, null, 1), 'utf8', function (err) {
                if (err) {
                    console.log("An error occured while writing JSON Object to File.");
                    return console.log(err);
                }
                console.log("JSON file has been saved.");
            });
        });
    });
}

getTickets().then(() => {
});

function findLink(url, componentName){
    return axios.get(url)
    .then(res =>{
        if(res !== undefined){
            const html = new JSDOM(res.data).window.document;
            if(url.includes('mongo')){
                if(url.includes('commit')){
                    let filteredHtml = html.documentElement.getElementsByClassName('file-header');
                    for(let i =0; i<filteredHtml.length; i++){
                        if(filteredHtml[i].textContent.includes('src')){
                            addToJSON(componentName, filteredHtml[i].textContent.split('src/')[1].split(' ')[0].split('\n')[0]);
                        }
                    }
                } else if(url.includes('/src') || url.includes('/jstests')){
                    if(url.includes('src/')){
                        addToJSON(componentName, 'src/'+url.split('src/')[1]);
                    }
                    if(url.includes('/jstests')){
                        addToJSON(componentName, 'jstests/'+url.split('jstests/')[1]);
                    }
                }
            }
        }
    }) 
    .catch(err => {
        //console.error('error: '+url);
    });
}

function addToJSON(componentName, path){
    if(directoryToComponents.find(element => element.name === componentName) === undefined){
        directoryToComponents.push({
            name: componentName,
            paths: [
                {
                    path: path
                }
            ]
        });
    } else {
        directoryToComponents[directoryToComponents.findIndex(element => element.name === componentName)].paths.push({path: path});
    }
}