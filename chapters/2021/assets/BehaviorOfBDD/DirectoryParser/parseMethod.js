const lineNumberByIndex = (index, string) => {
    // RegExp
    var line = 0,
        match,
        re = /(^)[\S\s]/gm;
    while (match = re.exec(string)) {
        if (match.index > index)
            break;
        line++;
    }
    return line;
}
module.exports = (str) => {
    const pattern = /(public|protected|private|static|[\r\t\f\v]) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(\{?|[^;])/g;
    let match;
    let result = [];
    while (match = pattern.exec(str)) {
        const index = lineNumberByIndex(match.index, str);
        // console.log(index);
        // console.log(str.split("\n")[index - 1])
        const methodLine = str.split("\n")[index - 1]
        const firstPart = methodLine.split("(")[0];
        const firstPart2 = firstPart.split(" ");
        //console.log(firstPart2);
        result.push({ name: firstPart2[firstPart2.length - 1], line: index });
    }
    return result;
}
