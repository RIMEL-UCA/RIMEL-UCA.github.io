WITH period AS 
( 
       SELECT * 
       FROM   `githubarchive.month.2019*` a
       
       WHERE repo.name IN (SELECT
            repo.name
        FROM
            `githubarchive.month.2019*`
        WHERE
            Json_extract_scalar(payload, '$.pull_request.base.repo.language') IS NOT NULL
            and Json_extract_scalar(payload, '$.pull_request.base.repo.language') = 'Java'

       )
       
       ),repo_stars AS 
( 
         SELECT   repo.id, 
                  count(DISTINCT actor.login)                     stars, 
                  approx_top_count(repo.NAME, 1)[OFFSET(0)].value repo_name 
         FROM     period 
         WHERE    type='WatchEvent' 
         GROUP BY 1 
         HAVING   stars>12), pushers_guess_emails_and_top_projects AS 
( 
       SELECT * , 
              regexp_replace(regexp_extract(email, r'@(.*)'), r'.*.ibm.com', 'ibm.com') domain 
       FROM   ( 
                       SELECT   actor.id , 
                                approx_top_count(actor.login,1)[OFFSET(0)].value                                               login , 
                                approx_top_count(json_extract_scalar(payload, '$.commits[0].author.email'),1)[OFFSET(0)].value email , 
                                count(*)                                                                                       c ,
                                array_agg(DISTINCT to_json_string(struct(b.repo_name,stars)))                                  repos
                       FROM     period a 
                       JOIN     repo_stars b 
                       ON       a.repo.id=b.id 
                       WHERE    type='PushEvent' 
                       GROUP BY 1 
                       HAVING   c>2 )) 
SELECT   * 
FROM     ( 
                SELECT domain , 
                       githubers , 
                       ( 
                              SELECT count(DISTINCT repo) 
                              FROM   unnest(repos) repo) repos_contributed_to , 
                       array 
                       ( 
                                SELECT   AS struct json_extract_scalar(repo, '$.repo_name') repo_name ,
                                         cast(json_extract_scalar(repo, '$.stars') AS int64)    stars ,
                                         count(*)                                               githubers_from_domain
                                FROM     unnest(repos) repo  
                                GROUP BY 1, 
                                         2 
                                HAVING   githubers_from_domain>1
                                ORDER BY stars DESC limit 100000 ) TOP , 
                       ( 
                              SELECT sum(cast(json_extract_scalar(repo, '$.stars') AS int64)) 
                              FROM   ( 
                                                     SELECT DISTINCT repo 
                                                     FROM            unnest(repos) repo)) sum_stars_projects_contributed_to
                FROM   ( 
                                SELECT   domain, 
                                         count(*) githubers, 
                                         array_concat_agg(array 
                                         ( 
                                                SELECT * 
                                                FROM   unnest(repos) repo)) repos 
                                FROM     pushers_guess_emails_and_top_projects 
                                WHERE    domain NOT IN unnest(split('gmail.com|users.noreply.github.com|qq.com|hotmail.com|163.com|me.com|googlemail.com|outlook.com|yahoo.com|web.de|iki.fi|foxmail.com|yandex.ru|126.com|protonmail.com', '|'))
                                GROUP BY 1 
                                HAVING   githubers > 12 ) 
                WHERE  ( 
                              SELECT max(githubers_from_domain) 
                              FROM   ( 
                                              SELECT   repo, 
                                                       count(*) githubers_from_domain 
                                              FROM     unnest(repos) repo 
                                              GROUP BY repo))>2 ) 
                          
ORDER BY githubers DESC
