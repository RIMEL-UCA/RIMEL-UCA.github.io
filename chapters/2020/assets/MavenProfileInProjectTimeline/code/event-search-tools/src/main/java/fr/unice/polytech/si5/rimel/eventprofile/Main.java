package fr.unice.polytech.si5.rimel.eventprofile;

import fr.unice.polytech.si5.rimel.eventprofile.commitanalyser.CommitAnalyser;
import fr.unice.polytech.si5.rimel.eventprofile.downloader.OptimizedDownloader;
import fr.unice.polytech.si5.rimel.eventprofile.model.Pom;
import fr.unice.polytech.si5.rimel.eventprofile.repo.RepositoryAnalyser;
import fr.unice.polytech.si5.rimel.eventprofile.searchdiff.ListDiffPomCommit;
import fr.unice.polytech.si5.rimel.eventprofile.technoanalyser.TechnoAnalyser;
import org.eclipse.jgit.api.Git;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.event.EventListener;
import org.springframework.data.neo4j.repository.config.EnableNeo4jRepositories;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@SpringBootApplication
@EnableNeo4jRepositories
public class Main {
    private Map<Pom, Integer> commitsNumbersByRepository = new HashMap<>();

    private static final  List<Pom> bannedPoms = new ArrayList<>();

    private static final String FILENAME = "stats-final-real.csv";

    private static final String BANNED_POM = "banned.csv";

    @Autowired
    private RepositoryAnalyser analyser;-

    @Autowired
    private CommitAnalyser commitAnalyser;

    @Autowired
    private TechnoAnalyser technoAnalyser;

    public static final String WORKSPACE = "./repositories/" ;

    public static void main(String[] args) {
        SpringApplication.run(fr.unice.polytech.si5.rimel.eventprofile.Main.class, args);
    }

    @EventListener(ApplicationReadyEvent.class)
    public void doSomethingAfterStartup() {
        try {
            getBannedPom();
            getPomsFromFile(407);
            int counter = 0;
            System.out.println(commitsNumbersByRepository.size());
            for (Map.Entry<Pom, Integer> pomIntegerEntry : commitsNumbersByRepository.entrySet()) {
                System.out.println("Repository number : " + counter + "/" + commitsNumbersByRepository.size());
                String repositoryURL = getRepoUrlFromFullName(pomIntegerEntry.getKey().getFullRepositoryName());

                // Clone repo
                Git repo = new OptimizedDownloader().openRepository(repositoryURL);

                // Analyse repo infos (number of branches)
                Pom pom = analyser.analyse(repo, pomIntegerEntry.getKey().getFullRepositoryName(), pomIntegerEntry.getKey().getPath());

                // List commits that modify the pom
                final List<String> commits = new ListDiffPomCommit().listDiff(repositoryURL, pomIntegerEntry.getKey().getPath());

                // Analyse retrieved commit by downloading poms
                commitAnalyser.analyseCommits(repo, commits, pomIntegerEntry.getKey().getPath(), pomIntegerEntry.getKey().getFullRepositoryName(), pom);

                // Analyse technos used in this project
                technoAnalyser.analyseTechno(repositoryURL, pomIntegerEntry.getKey().getPath());
                counter++;
            }

            final String repositoryName = "lnzttkx/spring-boot";
            final String repositoryURL = "https://github.com/lnzttkx/spring-boot.git";
            final String pomPath = "spring-boot-project/spring-boot-tools/spring-boot-gradle-plugin/pom.xml";


        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private String getRepoUrlFromFullName(String fullRepoName) {
        return "https://github.com/" + fullRepoName + ".git";
    }

    public void getBannedPom() {
        try (BufferedReader reader = new BufferedReader(new FileReader(BANNED_POM))) {
            String line = reader.readLine();
            while (line != null) {
                String[] split = line.split(",");
                Pom pom = Pom.builder().fullRepositoryName(split[0].replace("{fullRepositoryName:", "")).path(split[1].replace("path:", "").replace("}","")).build();
                bannedPoms.add(pom);
                line = reader.readLine();
            }
        } catch (FileNotFoundException ignored) {
            new File(FILENAME);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void getPomsFromFile(int commitThreshold) {
        try (BufferedReader reader = new BufferedReader(new FileReader(FILENAME))) {
            String line = reader.readLine();
            while (line != null) {
                String[] split = line.split(";");
                Pom pom = Pom.builder().fullRepositoryName(split[0]).path(split[1]).build();
                if (!bannedPoms.contains(pom)) {
                    int nbCommits = Integer.parseInt(split[2]);
                    if (nbCommits >= commitThreshold) {
                        commitsNumbersByRepository.put(pom, nbCommits);
                    }
                }
                line = reader.readLine();
            }
        } catch (FileNotFoundException ignored) {
            new File(FILENAME);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
