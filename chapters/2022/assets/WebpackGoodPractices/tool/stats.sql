-- COUNT project number > 2000 lignes de codes
SELECT COUNT(*)
FROM projects
WHERE rowsOfCode > 2000;


-- RepartitiON of stars by project categories
SELECT category, COUNT(p.stars) AS total
FROM categorizatiONs c LEFT JOIN projects p ON c.id = p.id
WHERE rowsOfCode > 2000
GROUP BY category;

-- Somme des Dependencies et moyennes des mauvaises Dependencies
SELECT SUM(dda.quantityOfDependencies + dda.quantityOfDevDependencies), AVG(dda.quantityOfWrONgDevDependencies/dda.quantityOfTargetDependencies) AS AVGOfWrONgPlaceForDepENDancies
FROM projects p
    LEFT JOIN categorizatiONs c ON p.id = c.id
    LEFT JOIN DevDependenciesAnalyzes dda ON dda.id = c.id
WHERE
    c.category != 'other';

-- Query ON % WrONgDevDependenciesEvolutiONByCategory
SELECT FLOOR(p.contributors/10)*10 AS contributorsScale, c.category, AVG(dda.quantityOfWrONgDevDependencies*100/dda.quantityOfTargetDependencies) AS AVGOfWrONgPlaceForDepENDancies
FROM projects p
    LEFT JOIN categorizatiONs c ON p.id = c.id
    LEFT JOIN DevDependenciesAnalyzes dda ON dda.id = c.id
WHERE
    c.category != 'other'
AND dda.quantityOfTargetDependencies >= 1
AND p.rowsOfCode > 2000
GROUP BY c.category,contributorsScale;



-- Query ON the biggest project (cONtributor or rowof code) % WrONgDevDependenciesEvolutiONByCategory
SELECT p.name, p.contributors, p.rowsOfCode, c.category, dda.quantityOfWrONgDevDependencies, dda.quantityOfTargetDependencies, (dda.quantityOfWrONgDevDependencies*100/dda.quantityOfTargetDependencies)  AS PourcentageOfWrONgPlaceForDepENDancies
FROM projects p
    LEFT JOIN categorizatiONs c ON p.id = c.id
    LEFT JOIN DevDependenciesAnalyzes dda ON dda.id = c.id
WHERE
    c.category != 'other'
-- ORDER BY p.contributors DESC;
ORDER BY p.rowsOfCode DESC;

-- Export for Excel wrONgUseOfDepENDancies
SELECT p.name, p.contributors, p.rowsOfCode, p.stars, c.category, (dda.quantityOfDependencies+dda.quantityOfDevDependencies) AS depdenciesTotal, dda.quantityOfWrONgDevDependencies, dda.quantityOfTargetDependencies , (dda.quantityOfWrONgDevDependencies*100/dda.quantityOfTargetDependencies)  AS PourcentageOfWrONgPlaceForDepENDancies
FROM projects p
    LEFT JOIN categorizatiONs c ON p.id = c.id
    LEFT JOIN DevDependenciesAnalyzes dda ON dda.id = c.id
WHERE
    c.category != 'other'
AND rowsOfCode > 2000
ORDER BY p.contributors DESC;


-- RepartitiON des projets en fONctiON du nombre de cONtributeur
SELECT t.RANGE AS [Nb CONtributeur], COUNT(*) AS [Nombre de projet]
FROM (
  SELECT cASe
    WHEN projects.contributors BETWEEN 0 AND 9 THEN ' 0-9'
    WHEN projects.contributors BETWEEN 10 AND 19 THEN '10-19'
    WHEN projects.contributors BETWEEN 20 AND 29 THEN '20-29'
    WHEN projects.contributors BETWEEN 30 AND 39 THEN '30-39'
    WHEN projects.contributors BETWEEN 40 AND 49 THEN '40-49'
    WHEN projects.contributors BETWEEN 50 AND 59 THEN '50-59'
    WHEN projects.contributors BETWEEN 60 AND 69 THEN '60-69'
    WHEN projects.contributors BETWEEN 70 AND 79 THEN '70-79'
    WHEN projects.contributors BETWEEN 80 AND 89 THEN '80-89'
    WHEN projects.contributors BETWEEN 90 AND 99 THEN '90-99'
    ELSE '99+' END AS RANGE
  FROM projects LEFT JOIN categorizatiONs c ON c.id = projects.id
    WHERE projects.rowsOfCode > 2000
    AND c.category != 'other' ) t
GROUP BY t.RANGE;

-- EsLint usage % GROUP BY category AND contributorsScale

SELECT FLOOR(p.contributors/10)*10 AS contributorsScale, c.category, COUNT(p.id) AS totalProject,
       SUM(CASE WHEN ela.quantityOfPlugins > 1 OR ela.quantityOfRules > 2 THEN 1 ELSE 0 END) AS quantityOfEslint,
       COUNT(p.id) / SUM(CASE WHEN ela.quantityOfPlugins > 1 OR ela.quantityOfRules > 2 THEN 1 ELSE 0 END) AS PourcentageOfEslintUsage
FROM projects p
    LEFT JOIN categorizatiONs c ON p.id = c.id
    LEFT JOIN EsLintAnalyzes ela ON ela.id = c.id
WHERE
    c.category != 'other'
AND p.rowsOfCode > 2000
GROUP BY c.category,contributorsScale;