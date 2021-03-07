#!/usr/bin/env sh
echo Step 1/2
echo This will find a link between files from code source and components.
echo Computed with 1 000 tickets by defaut
echo Please wait, it may take some time
node linkFilesToComponents.js
echo Done
echo Step 2/2
echo This will allow to verify our hypothesis about coupling.
echo Computed with 10 000 tickets by defaut.
echo Sometimes it is too much tickets, and you get an error depending on the conection.
echo You can retry using less tickets by changing maxResults value at the beggining of the findCoupling.js file.
echo Please wait, it may take some time.
node findCoupling.js
echo Done
echo Results are visible in result-10000.json