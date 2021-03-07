const express = require('express')
const cors = require('cors')
const app = express()
const morgan = require('morgan')
app.use(cors())
const bodyParser = require('body-parser')

app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*")
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept, Authorization")
    res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
    next()
});


app.use(bodyParser.urlencoded({extended: false}))
app.use(morgan('short'))



const parsing_router = require('./api/parsing.js')
const convert_router = require('./api/convertCSV.js')
app.use('/parsing',parsing_router);
app.use('/convert',convert_router);



app.listen(3001, () => {
    console.log("jira Api automation is up and listening on 3001...")
})