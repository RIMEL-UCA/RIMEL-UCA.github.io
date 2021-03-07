const express = require('express');
const  bodyParser = require('body-parser');
const stringify = require('csv-stringify');
const fs = require('fs');
const app =  express();
const parse_chart = require('../graph/piechart');
const parse_bar = require('../graph/barchart');


app.use(bodyParser.json());
app.use(bodyParser.urlencoded({
    extended: true
}));




const requesteService = require('../services/requetes.js');


app.get('/GeneralData',  async (req,res) => {
    try {
        let generalTabFilter = GeneralFiltersTab(req.body.ProjectName);

        for(let i in generalTabFilter) {

            let filter = generalTabFilter[i];

            let url = {
                "api": req.body.APIJira,
                "filter": filter
            };

            let stat = await requesteService.getStat(url);
            //Pie-chart Data
            let data_piechart = [];
            let columns_piechart = {
                component: 'label',
                nb: 'value'
            };

            for (let j in stat) {data_piechart.push([j, stat [j]]);}


            stringify(data_piechart, {header: true, columns: columns_piechart}, (err, output) => {
                if (err) throw err;
                let fileName = "./data/generalFilters/piechart/"+i+"_project_"+req.body.ProjectName+".csv";
                fs.writeFile(fileName, output, (err) => {
                    if (err) throw err;
                });
            });

            //Bar chart data
            let data_barchart = [];
            let columns_barchart = {
                component: 'key',
                nb: 'value'
            };

            for (let j in stat) {data_barchart.push([j, stat [j]]);}
            stringify(data_barchart, {header: true, columns: columns_barchart}, (err, output) => {
                if (err) throw err;
                let fileName = "./data/generalFilters/barchart/"+i+"_project_"+req.body.ProjectName+".csv";
                fs.writeFile(fileName, output, (err) => {
                    if (err) throw err;
                });
            });
        }
        parse_bar("generalFilters/barchart");
        parse_chart("generalFilters/piechart");
        await res.json("You can find the generated images in ./jira_api_automation/result/generalFilters");
    }
    catch (e) {
        await res.json({"error" : e})
    }
});

app.get('/specificDataComponents',  async (req,res) => {


    try {
        let SpecificFiltersTabFilter = SpecificFiltersTab1toMany(req.body.ProjectName,req.body.principalComponent,req.body.otherComponents);

        for(let i in SpecificFiltersTabFilter) {

            let filter = SpecificFiltersTabFilter[i];

            let url = {
                "api": req.body.APIJira,
                "filter": filter
            };

            let stat = await requesteService.getStat(url);

            // Pie Chart
            let data_piechart = [];
            let columns_piechart = {
                component: 'label',
                nb: 'value'
            };

            for (let j in stat) {data_piechart.push([j, stat [j]]);}


            stringify(data_piechart, {header: true, columns: columns_piechart}, (err, output) => {
                if (err) throw err;
                let fileName = "./data/specificFiltersOneToMany/piechart/"+i+"_project_"+req.body.ProjectName+".csv"
                fs.writeFile(fileName, output, (err) => {
                    if (err) throw err;
                });
            });

            // Bar Chart
            let data_barchart = [];
            let columns_barchart = {
                component: 'key',
                nb: 'value'
            };

            for (let j in stat) {data_barchart.push([j, stat [j]]);}


            stringify(data_barchart, {header: true, columns: columns_barchart}, (err, output) => {
                if (err) throw err;
                let fileName = "./data/specificFiltersOneToMany/barchart/"+i+"_project_"+req.body.ProjectName+".csv"
                fs.writeFile(fileName, output, (err) => {
                    if (err) throw err;
                });
            });

        }
        parse_bar("specificFiltersOneToMany/barchart");
        parse_chart("specificFiltersOneToMany/piechart");

        await res.json("You can find the generated images in ./jira_api_automation/result/specificFiltersOneToMany");
    }
    catch (e) {
        await res.json({"error" : e})
    }

});

app.get('/specificDataTwoComponents',  async (req,res) => {


    try {
        let SpecificFiltersTabFilter = SpecificFiltersTabOneToOne(req.body.ProjectName,req.body.firstComponent,req.body.secondComponent);

        for(let i in SpecificFiltersTabFilter) {

            let filter = SpecificFiltersTabFilter[i];

            let url = {
                "api": req.body.APIJira,
                "filter": filter
            };

            let stat = await requesteService.getStat(url);
            //pie_chart
            let data_piechart = [];
            let columns_piechart = {
                component: 'label',
                nb: 'value'
            };

            for (let j in stat) {data_piechart.push([j, stat [j]]);}


            stringify(data_piechart, {header: true, columns: columns_piechart}, (err, output) => {
                if (err) throw err;
                let fileName = "./data/specificFiltersOneToOne/piechart/"+i+"_project_"+req.body.ProjectName+".csv"
                fs.writeFile(fileName, output, (err) => {
                    if (err) throw err;
                });
            });

            //bar_chart
            let data_barchart = [];
            let columns_barchart = {
                component: 'key',
                nb: 'value'
            };

            for (let j in stat) {data_barchart.push([j, stat [j]]);}


            stringify(data_barchart, {header: true, columns: columns_barchart}, (err, output) => {
                if (err) throw err;
                let fileName = "./data/specificFiltersOneToOne/barchart/"+i+"_project_"+req.body.ProjectName+".csv" ;
                fs.writeFile(fileName, output, (err) => {
                    if (err) throw err;
                });
            });
        }
        parse_bar("specificFiltersOneToOne/barchart");
        parse_chart("specificFiltersOneToOne/piechart");
        await res.json("You can find the generated images in ./jira_api_automation/result/specificFiltersOneToOne");
    }
    catch (e) {
        await res.json({"error" : e})
    }
});





function GeneralFiltersTab(ProjectName) {
    let tab = {} ;
    let nb_bug_per_component = "project = "+ProjectName +" AND issuetype = Bug AND component is not EMPTY";
    let nb_bug_per_sprint    = "project = "+ProjectName +" AND issuetype = Bug AND Sprint is not EMPTY AND Sprint != Other";
    let nb_bug_2020 = "project = "+ProjectName +" AND issuetype = Bug AND component is not EMPTY AND created >= \"2020/01/01 00:00\" AND created < \"2020/12/31 23:59\"";
    let nb_bug_2019 = "project = "+ProjectName +" AND issuetype = Bug AND component is not EMPTY AND created >= \"2019/01/01 00:00\" AND created < \"2019/12/31 23:59\"";
    let closed_ticket_per_sprint = "project = "+ProjectName +" AND status = Closed AND Sprint is not EMPTY AND sprint != Other";
    let described_tickets = "project = "+ProjectName +" AND description is not EMPTY";
    let Tickets_with_no_description ="project = "+ProjectName +" AND description is EMPTY";
    let Improvement_2019 = "project = "+ProjectName +" AND issuetype = Improvement AND component is not EMPTY AND created >= \"2019/01/01 00:00\" AND created < \"2019/12/31 23:59\""
    let Improvement_2020 = "project = "+ProjectName +" AND issuetype = Improvement AND component is not EMPTY AND created >= \"2020/01/01 00:00\" AND created < \"2020/12/31 23:59\""
    let Improvement_NewFeature_2019 = "project = "+ProjectName +" AND issuetype in (Improvement, \"New Feature\") AND component is not EMPTY AND created >= \"2019/01/01 00:00\" AND created < \"2019/12/31 23:59\""
    let Improvement_NewFeature_2020 = "project = "+ProjectName +" AND issuetype in (Improvement, \"New Feature\") AND component is not EMPTY AND created >= \"2020/01/01 00:00\" AND created < \"2020/12/31 23:59\""
    let New_feature_2019 ="project = "+ProjectName +" AND issuetype = \"New Feature\" AND component is not EMPTY AND created >= \"2019/01/01 00:00\" AND created < \"2019/12/31 23:59\""
    let New_feature_2020 = "project = "+ProjectName +" AND issuetype = \"New Feature\" AND component is not EMPTY AND created >= \"2020/01/01 00:00\" AND created < \"2020/12/31 23:59\""
    let Bugs_P1_P2_per_component = "project = "+ProjectName +" AND issuetype = Bug AND priority = \"Critical - P2\" OR priority = \"Blocker - P1\" AND component is not EMPTY ";
    let Bugs_P3_P4_P5_per_component = "project = "+ProjectName +" AND issuetype = Bug AND priority = \"Minor - P4\" OR priority = P3 OR priority = \"Trivial - P5\"";
    let sharding_issues_number_type ="project = "+ProjectName +" AND component = Sharding";
    let Unsolved_tickets_with_high_priority = "project = "+ProjectName +" AND resolution = Unresolved AND priority = \"Blocker - P1\" OR priority = \"Critical - P2\" AND status != Closed";

    tab["nb_bug_per_component"] = nb_bug_per_component;
    tab["nb_bug_per_sprint"] = nb_bug_per_sprint;
    tab["nb_bug_2020"] = nb_bug_2020;
    tab["nb_bug_2019"] = nb_bug_2019;
    tab["closed_ticket_per_sprint"] = closed_ticket_per_sprint;
    tab["described_tickets"] = described_tickets;
    tab["Tickets_with_no_description"] = Tickets_with_no_description;
    tab["Improvement_2019"] = Improvement_2019;
    tab["Improvement_2020"] = Improvement_2020;
    tab["Improvement_NewFeature_2019"] = Improvement_NewFeature_2019;
    tab["Improvement_NewFeature_2020"] = Improvement_NewFeature_2020;
    tab["New_feature_2019"] = New_feature_2019;
    tab["New_feature_2020"] = New_feature_2020;
    tab["Bugs_P1_P2_per_component"] = Bugs_P1_P2_per_component;
    tab["Bugs_P3_P4_P5_per_component"] = Bugs_P3_P4_P5_per_component;
    tab["sharding_issues_number_type"] = sharding_issues_number_type;
    tab["Unsolved_tickets_with_high_priority"] = Unsolved_tickets_with_high_priority;

    return tab;

}

function SpecificFiltersTab1toMany(ProjectName,principalComponent,otherComponents) {
    let tab = {} ;
    let componentsNames = otherComponents[0];
    for (let i = 1 ; i<otherComponents.length ; i++){
        let a = otherComponents[i].split(" ")
        if(a.length == 2 ) componentsNames = componentsNames + ", \""+a[0]+" "+a[1]+"\"";
        else componentsNames = componentsNames + ", "+otherComponents[i]
    }
    let Bugs_ticket_between_components = "project = "+ProjectName +" AND component="+ principalComponent+ " AND component in ("+componentsNames+") AND type = Bug";
    let critical_blocker_bugs_shared_between_components  = "project = "+ProjectName +" AND component ="+ principalComponent + " AND component in ("+componentsNames+") AND type = Bug AND priority in (\"Critical - P2\", \"Blocker - P1\")"
    tab["Bugs_ticket_between_"+ProjectName+"_and_other_Components"] = Bugs_ticket_between_components;
    tab["critical_blocker_bugs_shared_between_"+ProjectName+"_Other_components"] = critical_blocker_bugs_shared_between_components;
    return tab;
}

function SpecificFiltersTabOneToOne(ProjectName,firstComponent,secondComponent) {
    let tab = {} ;

    let Link= "project = "+ProjectName +" AND component = "+firstComponent+" AND component = "+secondComponent;

    tab["Links_between_"+firstComponent+"_and_"+secondComponent] = Link;


    return tab;

}


module.exports = app;