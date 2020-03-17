package fr.unice.polytech.si5.rimel.mavenprofiles.categorize;

import fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model.Pom;
import fr.unice.polytech.si5.rimel.mavenprofiles.categorize.repository.IPomRepository;
import org.kohsuke.github.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class StatisticsAnalyser {
    private static final String FILENAME = "stats.csv";
    private Map<Pom, Integer> commitsNumbersByRepository = new HashMap<>();

    private static final String TOKEN = "";
    private static final String USERNAME = "";

    @Autowired
    private IPomRepository pomRepository;

    public void run() throws Exception {
        getPomsFromFile();
        List<Pom> poms = retrieveRepositories();
        /*//poms = new ArrayList<>();
        for (Map.Entry<Pom, Integer> pomIntegerEntry : commitsNumbersByRepository.entrySet()) {
            poms.add(pomIntegerEntry.getKey());
        }*/
        computeStatistics(poms);
    }

    private List<Pom> retrieveRepositories() {
        List<Pom> poms = new ArrayList<>();
        pomRepository.findAll().forEach(pom -> poms.add(Pom.builder().fullRepositoryName(pom.getFullRepositoryName()).path(pom.getPath()).build()));
        return poms;
    }

    private void computeStatistics(List<Pom> poms) {
        try {
            GitHub github = GitHub.connect(StatisticsAnalyser.USERNAME, StatisticsAnalyser.TOKEN);
            int countPom = 0;
            int totalPom = poms.size();
            for (Pom pom : poms) {
                String repositoryName = pom.getFullRepositoryName();
                countPom++;
                if (countPom < 0 || commitsNumbersByRepository.get(pom) != null)
                    continue;
                System.out.println(String.format("========== (%s/%s) - %s%% ==========", countPom, totalPom, (int) ((double) countPom / (double) totalPom * 10000) / 100.0));
                System.out.println(repositoryName);
                GHRepository repository;
                try {
                    repository = github.getRepository(repositoryName);
                } catch (GHFileNotFoundException | HttpException e) {
                    System.out.println("EXCEPTION:");
                    System.out.println(e.getMessage());
                    continue;
                }
                int countContributions;
                List<GHRepositoryStatistics.ContributorStats> contributors = repository.getStatistics().getContributorStats().asList();
                countContributions = contributors.stream().mapToInt(GHRepositoryStatistics.ContributorStats::getTotal).sum();
                commitsNumbersByRepository.put(pom, countContributions);
            }
            savePoms();
        } catch (Exception e) {
            e.printStackTrace();
            savePoms();
        }
    }

    private void writeStatistics(int contributors, int commits, int releases) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(FILENAME, true))) {
            writer.write(String.format("%s;%s;%s", contributors, commits, releases));
            writer.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void savePoms() {
        new File(FILENAME);
        for (Map.Entry<Pom, Integer> commitByPom : commitsNumbersByRepository.entrySet()) {
            Pom pom = commitByPom.getKey();
            int commits = commitByPom.getValue();
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(FILENAME, true))) {
                writer.write(String.format("%s;%s;%s", pom.getFullRepositoryName(), pom.getPath(), commits));
                writer.newLine();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void getPomsFromFile() {
        try (BufferedReader reader = new BufferedReader(new FileReader(FILENAME))) {
            String line = reader.readLine();
            while (line != null) {
                String[] split = line.split(";");
                Pom pom = Pom.builder().fullRepositoryName(split[0]).path(split[1]).build();
                commitsNumbersByRepository.put(pom, Integer.parseInt(split[2]));
                line = reader.readLine();
            }
        } catch (FileNotFoundException ignored) {
            new File(FILENAME);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
