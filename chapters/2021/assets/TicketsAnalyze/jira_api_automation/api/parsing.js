const express = require('express');
const  bodyParser = require('body-parser');
const app =  express();


app.use(bodyParser.json());
app.use(bodyParser.urlencoded({
    extended: true
}));

const requesteService = require('../services/requetes.js');

/******************** data ***********************/
app.put('/generalData',  async (req,res) => {

try{


    let data = await requesteService.getData(req.body);
    let issues = data.issues;
    let components={};

    issues.forEach(issue => {

            if(issue.fields.components[0] != undefined)
            {
                let a = components[issue.fields.components[0].name];
                if(a == undefined)  components[issue.fields.components[0].name] = 1 ;
                else  components[issue.fields.components[0].name]++;

            }

    });

    await res.json(components);

}

catch (e) {
    await res.json({"error" : e})
}

});



module.exports = app;