//Translate [{line: 1, value: true}] to [false, true]
const translateArrayObjectToArrayElement = (obj) => {

    const getValue = (index) => {
        for (const o of obj) {
            if (o.nr === index) return o.c;
        }
        return null;
    }

    const max = Math.max(...obj.map(e => e.nr))
    const arr = []
    for (let i = 0; i < max; i++) {
        let value = getValue(i);
        if (value === null) value = false;
        arr.push(value)
    }

    return arr;
}

module.exports.parse = (str) => {
    const base = JSON.parse(str).report.package.map(e => { return { name: e.name, data: e.sourcefile } })

    return base.reduce((acc, value) => Array.isArray(value.data) ? [...acc, ...value.data] : [...acc, value.data], []).map(e => {
        if (!e.line) {
            return undefined;
        } else if (!Array.isArray(e.line)) {
            return {
                name: e.name,
                line: [{ nr: parseInt(e.line.nr), c: e.line.cb > 0 || (e.line.mi == 0 && e.line.mb == 0) ? 1 : 0 }],
            }
        }
        return {
            name: e.name, line: e.line.map(e => {
                return { nr: parseInt(e.nr), c: e.cb > 0 || (e.mi == 0 && e.mb == 0) ? 1 : 0 }
            })
        }
    }
    ).filter(e => e);
}

// out-X_error-map-undefined-coverage-parser