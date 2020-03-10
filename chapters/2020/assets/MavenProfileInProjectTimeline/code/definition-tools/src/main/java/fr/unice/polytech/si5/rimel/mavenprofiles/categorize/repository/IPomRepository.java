package fr.unice.polytech.si5.rimel.mavenprofiles.categorize.repository;

import fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model.Pom;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface IPomRepository extends Neo4jRepository<Pom, String> {

	Optional<Pom> findByFullRepositoryNameAndPath(String fullRepositoryName, String path);

	Optional<Pom> findByFullRepositoryName(String fullRepositoryName);
}
