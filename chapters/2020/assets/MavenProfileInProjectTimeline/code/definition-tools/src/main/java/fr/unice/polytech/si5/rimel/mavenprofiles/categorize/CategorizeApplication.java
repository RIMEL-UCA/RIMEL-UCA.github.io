package fr.unice.polytech.si5.rimel.mavenprofiles.categorize;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.data.neo4j.repository.config.EnableNeo4jRepositories;

@SpringBootApplication
@EnableNeo4jRepositories
public class CategorizeApplication {

	@Autowired
	private Analyzer analyzer;

	@Autowired
	private StatisticsAnalyser statAnalyser;

	public static void main(String[] args) {
		SpringApplication.run(CategorizeApplication.class, args);
	}

	@EventListener(ApplicationReadyEvent.class)
	public void doSomethingAfterStartup() {
		try {
			analyzer.run();
			//statAnalyser.run();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
