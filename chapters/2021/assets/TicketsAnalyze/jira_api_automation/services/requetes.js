const axios = require('axios');

module.exports = {





    getStat: function (url) {

        return new Promise((resolve, reject) => {

            axios.put('http://localhost:3001/parsing/generalData',url).then(result => resolve(result.data))
        })
    },


    getData: function (url) {
        let request = url.api+"?jql="+url.filter;
        return new Promise((resolve, reject) => {
            axios.get(request).then(result => resolve(result.data));
        })
    },
};